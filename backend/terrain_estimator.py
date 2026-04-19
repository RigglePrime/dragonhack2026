from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import typer
import yfinance as yf
from rasterio.windows import Window

VALID_DN_MIN = 0.0
VALID_DN_MAX = 250.0

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class MatchResult:
    distance: float
    score: float
    row: float
    col: float
    heading_deg: float


@dataclass(frozen=True)
class CoordinateCandidate:
    rank: int
    score: float
    row: float
    col: float
    heading_deg: float
    projected_x: float
    projected_y: float
    lat: float
    lon: float
    route_latlon: list[tuple[float, float]]
    terrain_profile: list[float]


def dn_to_slope_degrees(dn_values: np.ndarray) -> np.ndarray:
    # EUDEM formula: slope[degrees] = acos(DN / 250) * 180 / pi
    scaled = np.clip(dn_values / VALID_DN_MAX, -1.0, 1.0)
    return np.degrees(np.arccos(scaled))


class TerrainMap:
    def __init__(self, tiff_path: Path) -> None:
        self.path = tiff_path
        self.dataset = rasterio.open(tiff_path)

        self.width = self.dataset.width
        self.height = self.dataset.height
        self.transform = self.dataset.transform
        self.epsg = self.dataset.crs.to_epsg() if self.dataset.crs is not None else None

        self.scale_x = abs(float(self.transform.a))
        self.scale_y = abs(float(self.transform.e))
        if self.scale_x <= 0.0 or self.scale_y <= 0.0:
            raise ValueError("Invalid GeoTIFF transform resolution.")

        self._transformer = None
        if self.epsg is not None:
            try:
                from pyproj import Transformer

                self._transformer = Transformer.from_crs(
                    f"EPSG:{self.epsg}",
                    "EPSG:4326",
                    always_xy=True,
                )
            except Exception:
                self._transformer = None

    def close(self) -> None:
        self.dataset.close()

    def pixel_to_projected(self, row: float, col: float) -> tuple[float, float]:
        x, y = self.dataset.xy(row, col)
        return float(x), float(y)

    def projected_to_wgs84(self, x: float, y: float) -> Optional[tuple[float, float]]:
        if self._transformer is None:
            return None
        lon, lat = self._transformer.transform(x, y)
        return float(lat), float(lon)

    def _line_offsets(
        self,
        center_row: float,
        center_col: float,
        heading_deg: float,
        spacing_px: float,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        theta = math.radians(heading_deg)
        dcol = math.sin(theta)
        drow = -math.cos(theta)

        offsets = (np.arange(n_samples, dtype=np.float64) - (n_samples - 1) / 2.0) * spacing_px
        rows = center_row + offsets * drow
        cols = center_col + offsets * dcol
        return rows, cols

    def _validate_bounds(self, rows: np.ndarray, cols: np.ndarray) -> bool:
        return not (
            rows.min() < 1
            or cols.min() < 1
            or rows.max() >= self.height - 2
            or cols.max() >= self.width - 2
        )

    def sample_line(
        self,
        center_row: float,
        center_col: float,
        heading_deg: float,
        spacing_px: float,
        n_samples: int,
    ) -> Optional[np.ndarray]:
        if n_samples < 2:
            return None

        rows, cols = self._line_offsets(center_row, center_col, heading_deg, spacing_px, n_samples)
        if not self._validate_bounds(rows, cols):
            return None

        row0 = np.floor(rows).astype(np.int64)
        col0 = np.floor(cols).astype(np.int64)
        row1 = row0 + 1
        col1 = col0 + 1

        top = int(row0.min())
        bottom = int(row1.max())
        left = int(col0.min())
        right = int(col1.max())

        window = Window(
            col_off=left,
            row_off=top,
            width=(right - left + 1),
            height=(bottom - top + 1),
        )
        patch = self.dataset.read(1, window=window, out_dtype=np.float32)

        local_r0 = row0 - top
        local_c0 = col0 - left
        local_r1 = row1 - top
        local_c1 = col1 - left

        fr = rows - row0
        fc = cols - col0

        v00 = patch[local_r0, local_c0]
        v01 = patch[local_r0, local_c1]
        v10 = patch[local_r1, local_c0]
        v11 = patch[local_r1, local_c1]

        dn_values = (
            v00 * (1.0 - fr) * (1.0 - fc)
            + v01 * (1.0 - fr) * fc
            + v10 * fr * (1.0 - fc)
            + v11 * fr * fc
        ).astype(np.float64)

        if np.any(~np.isfinite(dn_values)):
            return None
        if np.any(dn_values < VALID_DN_MIN) or np.any(dn_values > VALID_DN_MAX):
            return None

        return dn_to_slope_degrees(dn_values)

    def route_latlon(
        self,
        center_row: float,
        center_col: float,
        heading_deg: float,
        spacing_px: float,
        n_points: int,
    ) -> list[tuple[float, float]]:
        rows, cols = self._line_offsets(center_row, center_col, heading_deg, spacing_px, n_points)
        if not self._validate_bounds(rows, cols):
            return []

        route: list[tuple[float, float]] = []
        for row, col in zip(rows, cols):
            x, y = self.pixel_to_projected(float(row), float(col))
            latlon = self.projected_to_wgs84(x, y)
            if latlon is not None:
                route.append(latlon)
        return route


def normalized(arr: np.ndarray) -> Optional[np.ndarray]:
    mean = arr.mean()
    std = arr.std()
    if std < 1e-6:
        return None
    return (arr - mean) / std


def profile_distance(profile: np.ndarray, terrain: np.ndarray) -> float:
    pz = normalized(profile)
    tz = normalized(terrain)
    if pz is None or tz is None:
        return float("inf")

    primary = float(np.linalg.norm(pz - tz) / math.sqrt(pz.size))

    pd = np.diff(profile)
    td = np.diff(terrain)
    pdz = normalized(pd)
    tdz = normalized(td)
    if pdz is None or tdz is None:
        return primary

    derivative = float(np.linalg.norm(pdz - tdz) / math.sqrt(pdz.size))
    return 0.7 * primary + 0.3 * derivative


def evaluate_candidate(
    terrain_map: TerrainMap,
    profile: np.ndarray,
    row: float,
    col: float,
    heading_deg: float,
    spacing_px: float,
) -> tuple[float, float]:
    line = terrain_map.sample_line(row, col, heading_deg, spacing_px, profile.size)
    if line is None:
        return float("inf"), -2.0

    distance = profile_distance(profile, line)
    if not np.isfinite(distance):
        return float("inf"), -2.0

    # Keep score for compatibility with current frontend/output format.
    score = 1.0 / (1.0 + distance)
    return distance, score


def search_tercom(
    terrain_map: TerrainMap,
    profile: np.ndarray,
    spacing_m: float,
    headings_deg: np.ndarray,
    random_samples: int,
    top_k: int,
    refine_iters: int,
    refine_step_px: float,
    rng: np.random.Generator,
) -> list[MatchResult]:
    _ = refine_iters, refine_step_px  # Kept for API compatibility; current algorithm is single-pass.

    spacing_px = spacing_m / terrain_map.scale_x
    if spacing_px <= 0:
        raise ValueError("spacing_m must be positive.")

    half_extent = int(math.ceil((profile.size - 1) * spacing_px / 2.0)) + 2
    if half_extent * 2 >= min(terrain_map.width, terrain_map.height):
        raise ValueError("Profile extent is too large for this map.")

    placed_locations = max(1, int(random_samples))
    selected_count = max(1, int(top_k))
    candidates: list[MatchResult] = []

    row_low = half_extent
    row_high = terrain_map.height - half_extent
    col_low = half_extent
    col_high = terrain_map.width - half_extent

    for _ in range(placed_locations):
        row = float(rng.integers(row_low, row_high))
        col = float(rng.integers(col_low, col_high))

        best_local: Optional[MatchResult] = None
        for heading in headings_deg:
            distance, score = evaluate_candidate(
                terrain_map,
                profile,
                row,
                col,
                float(heading),
                spacing_px,
            )

            current = MatchResult(
                distance=distance,
                score=score,
                row=row,
                col=col,
                heading_deg=float(heading),
            )
            if best_local is None or current.distance < best_local.distance:
                best_local = current

        if best_local is not None and np.isfinite(best_local.distance):
            candidates.append(best_local)

    candidates.sort(key=lambda item: item.distance)
    closest = candidates[:selected_count]

    return closest


def estimate_candidates(
    profile: np.ndarray,
    map_path: Path,
    top_n: int = 10,
    spacing_m: float = 25.0,
    headings: int = 12,
    random_samples: int = 300,
    top_k: int = 20,
    refine_iters: int = 2,
    refine_step_px: float = 120.0,
    seed: int = 42,
) -> list[CoordinateCandidate]:
    _ = random_samples, top_k  # Kept for compatibility with existing service/frontend payload.

    clean_profile = np.asarray(profile, dtype=np.float64)
    clean_profile = clean_profile[np.isfinite(clean_profile)]
    if clean_profile.size < 8:
        raise ValueError("Profile must contain at least 8 valid points.")

    headings_deg = np.linspace(0.0, 360.0, headings, endpoint=False)
    rng = np.random.default_rng(seed)

    # Requested behavior: place 100 locations and keep the 10 closest.
    placed_locations = 100
    selected_count = min(max(1, int(top_n)), 10)

    terrain_map = TerrainMap(map_path)
    try:
        matches = search_tercom(
            terrain_map=terrain_map,
            profile=clean_profile,
            spacing_m=spacing_m,
            headings_deg=headings_deg,
            random_samples=placed_locations,
            top_k=max(selected_count, 10),
            refine_iters=refine_iters,
            refine_step_px=refine_step_px,
            rng=rng,
        )

        spacing_px = spacing_m / terrain_map.scale_x
        candidates: list[CoordinateCandidate] = []
        for rank, match in enumerate(matches[:selected_count], start=1):
            x, y = terrain_map.pixel_to_projected(match.row, match.col)
            latlon = terrain_map.projected_to_wgs84(x, y)
            if latlon is None:
                continue

            lat, lon = latlon
            route = terrain_map.route_latlon(
                center_row=match.row,
                center_col=match.col,
                heading_deg=match.heading_deg,
                spacing_px=spacing_px,
                n_points=int(clean_profile.size),
            )
            terrain_profile=terrain_map.sample_line(match.row, match.col, match.heading_deg, spacing_px, clean_profile.size, )

            candidates.append(
                CoordinateCandidate(
                    rank=rank,
                    score=match.score,
                    row=match.row,
                    col=match.col,
                    heading_deg=match.heading_deg,
                    projected_x=x,
                    projected_y=y,
                    lat=lat,
                    lon=lon,
                    route_latlon=route,
                    terrain_profile=terrain_profile.tolist() if terrain_profile is not None else [],
                )
            )
        return candidates
    finally:
        terrain_map.close()


def fetch_stock_close_profile(symbol: str, window: str) -> np.ndarray:
    config = {
        "1d": {"period": "1d", "interval": "5m"},
        "1w": {"period": "5d", "interval": "30m"},
        "1mo": {"period": "1mo", "interval": "1d"},
    }
    if window not in config:
        raise ValueError("window must be one of: 1d, 1w, 1mo")

    history = yf.Ticker(symbol).history(
        period=config[window]["period"],
        interval=config[window]["interval"],
        auto_adjust=False,
    )
    if history.empty:
        raise ValueError(f"No Yahoo Finance data found for symbol '{symbol}'.")

    closes = history["Close"].dropna().to_numpy(dtype=np.float64)
    if closes.size < 8:
        raise ValueError(f"Not enough close-price points for symbol '{symbol}'.")
    return closes


@app.command("estimate-stock")
def estimate_stock_cli(
    symbol: str = typer.Option(..., "--symbol", help="Ticker symbol, e.g. AAPL"),
    window: str = typer.Option("1mo", "--window", help="History window: 1d, 1w, or 1mo."),
    map_path: Path = typer.Option(
        Path("eud_cp_slop/eudem_slop_3035_europe.tif"),
        "--map",
        help="Path to EUDEM GeoTIFF.",
    ),
    top_n: int = typer.Option(10, "--top-n", help="How many candidates to print."),
    spacing_m: float = typer.Option(25.0, "--spacing-m"),
    headings: int = typer.Option(12, "--headings"),
    random_samples: int = typer.Option(300, "--samples"),
    top_k: int = typer.Option(20, "--top-k"),
    refine_iters: int = typer.Option(2, "--refine-iters"),
    refine_step_px: float = typer.Option(120.0, "--refine-step-px"),
    seed: int = typer.Option(42, "--seed"),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    profile = fetch_stock_close_profile(symbol=symbol.strip().upper(), window=window)
    candidates = estimate_candidates(
        profile=profile,
        map_path=map_path,
        top_n=top_n,
        spacing_m=spacing_m,
        headings=headings,
        random_samples=random_samples,
        top_k=top_k,
        refine_iters=refine_iters,
        refine_step_px=refine_step_px,
        seed=seed,
    )

    if as_json:
        payload: list[dict] = []
        for candidate in candidates:
            item = asdict(candidate)
            item["route_latlon"] = [
                {"lat": lat, "lon": lon} for lat, lon in candidate.route_latlon
            ]
            payload.append(item)
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Top {len(candidates)} candidates for {symbol.upper()} ({window})")
    for candidate in candidates:
        typer.echo(
            f"#{candidate.rank} score={candidate.score:.4f} lat={candidate.lat:.6f} "
            f"lon={candidate.lon:.6f} heading={candidate.heading_deg:.2f}"
        )


if __name__ == "__main__":
    app()
