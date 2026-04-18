from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import typer
import rasterio
from rasterio.windows import Window


@dataclass(frozen=True)
class MatchResult:
    score: float
    row: float
    col: float
    heading_deg: float


app = typer.Typer(add_completion=False)

ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
VALID_DN_MIN = 0.0
VALID_DN_MAX = 250.0


class TerrainMap:
    def __init__(self, tiff_path: Path) -> None:
        if rasterio is None:
            raise RuntimeError(
                "Missing dependency: rasterio. Install with 'python -m pip install rasterio'."
            )

        self.path = tiff_path
        self.dataset = rasterio.open(tiff_path)

        self.width = self.dataset.width
        self.height = self.dataset.height
        self.transform = self.dataset.transform
        self.epsg = self.dataset.crs.to_epsg() if self.dataset.crs is not None else None
        self.nodata = self.dataset.nodata

        self.scale_x = abs(float(self.transform.a))
        self.scale_y = abs(float(self.transform.e))
        if self.scale_x <= 0.0 or self.scale_y <= 0.0:
            raise ValueError("Invalid GeoTIFF transform resolution.")

    def close(self) -> None:
        self.dataset.close()

    def pixel_to_projected(self, row: float, col: float) -> tuple[float, float]:
        x, y = self.dataset.xy(row, col)
        return float(x), float(y)

    def projected_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        row, col = self.dataset.index(x, y)
        return float(row), float(col)

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

        theta = math.radians(heading_deg)
        dcol = math.sin(theta)
        drow = -math.cos(theta)

        offsets = (np.arange(n_samples, dtype=np.float64) - (n_samples - 1) / 2.0) * spacing_px
        rows = center_row + offsets * drow
        cols = center_col + offsets * dcol

        if (
            rows.min() < 1
            or cols.min() < 1
            or rows.max() >= self.height - 2
            or cols.max() >= self.width - 2
        ):
            return None

        row0 = np.floor(rows).astype(np.int64)
        col0 = np.floor(cols).astype(np.int64)
        row1 = row0 + 1
        col1 = col0 + 1

        top = int(row0.min())
        bottom = int(row1.max())
        left = int(col0.min())
        right = int(col1.max())

        # Windowed read keeps memory bounded to only the needed patch.
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

        values = (
            v00 * (1.0 - fr) * (1.0 - fc)
            + v01 * (1.0 - fr) * fc
            + v10 * fr * (1.0 - fc)
            + v11 * fr * fc
        )

        dn_values = values.astype(np.float64)
        if np.any(~np.isfinite(dn_values)):
            return None
        if np.any(dn_values < VALID_DN_MIN) or np.any(dn_values > VALID_DN_MAX):
            return None

        return dn_to_slope_degrees(dn_values)


def load_profile(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        profile = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    else:
        data = np.genfromtxt(path, delimiter=",", dtype=np.float64)
        if data.ndim == 0:
            profile = np.asarray([float(data)], dtype=np.float64)
        elif data.ndim == 1:
            profile = data.astype(np.float64)
        else:
            profile = data[:, -1].astype(np.float64)

    profile = profile[np.isfinite(profile)]
    if profile.size < 8:
        raise ValueError("Profile must contain at least 8 valid samples.")
    return profile


def dn_to_slope_degrees(dn_values: np.ndarray) -> np.ndarray:
    # EUDEM: slope[degrees] = acos(DN / 250) * 180 / pi
    dn_clamped = np.clip(dn_values / VALID_DN_MAX, -1.0, 1.0)
    return np.degrees(np.arccos(dn_clamped))


def load_alpha_vantage_api_key(env_file: Path) -> str:
    for env_key in ("ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY", "API_KEY"):
        env_value = os.getenv(env_key)
        if env_value:
            return env_value.strip()

    if not env_file.exists():
        raise ValueError(f"API key file not found: {env_file}")

    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        values[key.strip()] = raw_value.strip().strip('"').strip("'")

    for file_key in ("ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY", "API_KEY"):
        if file_key in values and values[file_key]:
            return values[file_key]

    raise ValueError(
        "No AlphaVantage API key found. Add ALPHAVANTAGE_API_KEY or API_KEY to .env."
    )


def _extract_stock_value(entry: dict[str, str], field: str) -> float:
    suffix_by_field = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjusted_close": "adjusted close",
        "volume": "volume",
    }
    if field not in suffix_by_field:
        raise ValueError(f"Unsupported stock field: {field}")

    wanted_suffix = suffix_by_field[field]
    for key, value in entry.items():
        if key.lower().endswith(wanted_suffix):
            return float(value)

    if field == "adjusted_close":
        for key, value in entry.items():
            if key.lower().endswith("close"):
                return float(value)

    raise ValueError(f"Field '{field}' is not available in AlphaVantage response entry.")


def fetch_stock_profile(
    symbol: str,
    points: int,
    field: str,
    api_key: str,
    base_url: str = ALPHAVANTAGE_URL,
) -> np.ndarray:
    field = field.lower().strip()
    if points > 100:
        raise ValueError(
            "AlphaVantage free daily endpoint returns up to 100 points. Use --stock-points <= 100."
        )

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": api_key,
    }
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "Error Message" in payload:
        raise ValueError(f"AlphaVantage error: {payload['Error Message']}")
    if "Information" in payload:
        raise RuntimeError(f"AlphaVantage information message: {payload['Information']}")
    if "Note" in payload:
        raise RuntimeError(f"AlphaVantage rate-limit note: {payload['Note']}")

    ts_key = None
    for key in payload.keys():
        if "Time Series" in key:
            ts_key = key
            break
    if ts_key is None:
        raise ValueError("Could not find Time Series data in AlphaVantage response.")

    time_series = payload[ts_key]
    if not isinstance(time_series, dict) or len(time_series) == 0:
        raise ValueError("AlphaVantage returned empty time series.")

    ordered_dates = sorted(time_series.keys())
    samples = np.asarray(
        [_extract_stock_value(time_series[date], field) for date in ordered_dates],
        dtype=np.float64,
    )

    finite = samples[np.isfinite(samples)]
    if finite.size < points:
        raise ValueError(
            f"Requested {points} points for {symbol}, but only {finite.size} valid points are available."
        )
    return finite[-points:]


def normalized(arr: np.ndarray) -> Optional[np.ndarray]:
    mean = arr.mean()
    std = arr.std()
    if std < 1e-6:
        return None
    return (arr - mean) / std


def tercom_score(
    profile: np.ndarray,
    terrain: np.ndarray,
    nodata_min: Optional[float],
) -> float:
    if nodata_min is not None and np.any(terrain >= nodata_min):
        return -2.0

    pz = normalized(profile)
    tz = normalized(terrain)
    if pz is None or tz is None:
        return -2.0

    corr = float(np.mean(pz * tz))

    pd = np.diff(profile)
    td = np.diff(terrain)
    pdz = normalized(pd)
    tdz = normalized(td)
    if pdz is None or tdz is None:
        return corr

    diff_corr = float(np.mean(pdz * tdz))
    return 0.7 * corr + 0.3 * diff_corr


def evaluate_candidate(
    terrain_map: TerrainMap,
    profile: np.ndarray,
    row: float,
    col: float,
    heading_deg: float,
    spacing_px: float,
    nodata_min: Optional[float],
) -> float:
    line = terrain_map.sample_line(row, col, heading_deg, spacing_px, profile.size)
    if line is None:
        return -2.0
    return tercom_score(
        profile,
        line,
        nodata_min=nodata_min,
    )


def search_tercom(
    terrain_map: TerrainMap,
    profile: np.ndarray,
    spacing_m: float,
    headings_deg: np.ndarray,
    random_samples: int,
    top_k: int,
    refine_iters: int,
    refine_step_px: float,
    nodata_min: Optional[float],
    rng: np.random.Generator,
) -> list[MatchResult]:
    spacing_px = spacing_m / terrain_map.scale_x
    if spacing_px <= 0:
        raise ValueError("spacing_m must be positive.")

    half_extent = int(math.ceil((profile.size - 1) * spacing_px / 2.0)) + 2
    if half_extent * 2 >= min(terrain_map.width, terrain_map.height):
        raise ValueError("Profile extent is too large for this map.")

    heap: list[tuple[float, float, float, float]] = []

    row_low = half_extent
    row_high = terrain_map.height - half_extent
    col_low = half_extent
    col_high = terrain_map.width - half_extent

    for _ in range(random_samples):
        row = float(rng.integers(row_low, row_high))
        col = float(rng.integers(col_low, col_high))
        for heading in headings_deg:
            score = evaluate_candidate(
                terrain_map,
                profile,
                row,
                col,
                float(heading),
                spacing_px,
                nodata_min,
            )
            if len(heap) < top_k:
                heapq.heappush(heap, (score, row, col, float(heading)))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, row, col, float(heading)))

    best = sorted(heap, key=lambda x: x[0], reverse=True)

    refined: list[MatchResult] = []
    for score, row, col, heading in best:
        cur_score = score
        cur_row = row
        cur_col = col
        cur_heading = heading

        step = refine_step_px
        heading_step = 8.0
        for _ in range(refine_iters):
            candidates: list[tuple[float, float, float, float]] = []
            for dr in (-step, 0.0, step):
                for dc in (-step, 0.0, step):
                    for dh in (-heading_step, 0.0, heading_step):
                        rr = cur_row + dr
                        cc = cur_col + dc
                        hh = (cur_heading + dh) % 360.0
                        s = evaluate_candidate(
                            terrain_map,
                            profile,
                            rr,
                            cc,
                            hh,
                            spacing_px,
                            nodata_min,
                        )
                        candidates.append((s, rr, cc, hh))

            best_local = max(candidates, key=lambda x: x[0])
            if best_local[0] >= cur_score:
                cur_score, cur_row, cur_col, cur_heading = best_local

            step = max(1.0, step / 2.0)
            heading_step = max(0.5, heading_step / 2.0)

        refined.append(
            MatchResult(
                score=cur_score,
                row=cur_row,
                col=cur_col,
                heading_deg=cur_heading,
            )
        )

    refined.sort(key=lambda m: m.score, reverse=True)
    deduped: list[MatchResult] = []
    for item in refined:
        keep = True
        for existing in deduped:
            dist = math.hypot(item.row - existing.row, item.col - existing.col)
            heading_delta = abs(((item.heading_deg - existing.heading_deg + 180.0) % 360.0) - 180.0)
            if dist < 8.0 and heading_delta < 1.0:
                keep = False
                break
        if keep:
            deduped.append(item)
    return deduped


def projected_to_wgs84(x: float, y: float, src_epsg: Optional[int]) -> Optional[tuple[float, float]]:
    if src_epsg is None:
        return None
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(f"EPSG:{src_epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return float(lat), float(lon)
    except Exception:
        return None


def format_match(rank: int, match: MatchResult, terrain_map: TerrainMap) -> str:
    x, y = terrain_map.pixel_to_projected(match.row, match.col)
    latlon = projected_to_wgs84(x, y, terrain_map.epsg)
    base = (
        f"#{rank} score={match.score:.5f} pixel(row={match.row:.1f}, col={match.col:.1f}) "
        f"heading={match.heading_deg:.2f}deg projected=({x:.2f}, {y:.2f})"
    )
    if latlon is None:
        return base
    lat, lon = latlon
    return f"{base} lat={lat:.6f} lon={lon:.6f}"


def run_self_test(
    terrain_map: TerrainMap,
    spacing_m: float,
    heading_deg: Optional[float],
    headings: int,
    samples: int,
    topk: int,
    refine_iters: int,
    refine_step_px: float,
    nodata_min: Optional[float],
    seed: int,
    self_test_points: int,
    self_test_noise: float,
) -> int:
    rng = np.random.default_rng(seed)
    spacing_px = spacing_m / terrain_map.scale_x
    half_extent = int(math.ceil((self_test_points - 1) * spacing_px / 2.0)) + 10

    true_row = float(rng.integers(half_extent, terrain_map.height - half_extent))
    true_col = float(rng.integers(half_extent, terrain_map.width - half_extent))
    true_heading = float(rng.uniform(0.0, 360.0))

    clean = terrain_map.sample_line(
        true_row,
        true_col,
        true_heading,
        spacing_px,
        self_test_points,
    )
    if clean is None:
        typer.echo("Self-test failed to generate synthetic profile.")
        return 2

    noisy = clean + rng.normal(0.0, self_test_noise, size=clean.shape)
    headings = (
        np.array([true_heading])
        if heading_deg is not None
        else np.linspace(0, 360, headings, endpoint=False)
    )
    results = search_tercom(
        terrain_map=terrain_map,
        profile=noisy,
        spacing_m=spacing_m,
        headings_deg=headings,
        random_samples=samples,
        top_k=topk,
        refine_iters=refine_iters,
        refine_step_px=refine_step_px,
        nodata_min=nodata_min,
        rng=rng,
    )

    if not results:
        typer.echo("No match candidates found.")
        return 3

    best = results[0]
    pixel_error = math.hypot(best.row - true_row, best.col - true_col)
    meter_error = pixel_error * terrain_map.scale_x
    heading_error = abs(((best.heading_deg - true_heading + 180.0) % 360.0) - 180.0)

    typer.echo("TERCOM self-test")
    typer.echo(f"true:  row={true_row:.2f} col={true_col:.2f} heading={true_heading:.2f}deg")
    typer.echo(f"best:  {format_match(1, best, terrain_map)}")
    typer.echo(f"error: pixels={pixel_error:.2f} meters={meter_error:.2f} heading_deg={heading_error:.2f}")
    return 0


@app.command()
def main(
    map_path: Path = typer.Option(
        Path("eud_cp_slop/eudem_slop_3035_europe.tif"),
        "--map",
        help="Path to the GeoTIFF terrain map.",
    ),
    profile: Optional[Path] = typer.Option(
        None,
        "--profile",
        help="CSV/TXT/NPY profile file (1D elevation samples).",
    ),
    stock_symbol: Optional[str] = typer.Option(
        None,
        "--stock-symbol",
        help="Stock symbol (e.g., AAPL). If set, profile comes from AlphaVantage history.",
    ),
    stock_points: int = typer.Option(
        90,
        "--stock-points",
        help="Number of historical stock points to use as the profile (max 100 on free AlphaVantage).",
    ),
    stock_field: str = typer.Option(
        "close",
        "--stock-field",
        help="Stock field to use: open, high, low, close, adjusted_close, volume.",
    ),
    alpha_env_file: Path = typer.Option(
        Path(".env"),
        "--alpha-env-file",
        help="Path to .env containing ALPHAVANTAGE_API_KEY or API_KEY.",
    ),
    spacing_m: float = typer.Option(
        25.0,
        "--spacing-m",
        help="Distance in meters between consecutive profile samples.",
    ),
    heading_deg: Optional[float] = typer.Option(
        None,
        "--heading-deg",
        help="Known heading in degrees (clockwise from north). If omitted, headings are searched.",
    ),
    headings: int = typer.Option(
        36,
        "--headings",
        help="Number of headings to test when heading is unknown.",
    ),
    samples: int = typer.Option(
        1200,
        "--samples",
        help="Number of random positions in the coarse search.",
    ),
    topk: int = typer.Option(
        20,
        "--topk",
        help="Number of best coarse candidates kept for local refinement.",
    ),
    refine_iters: int = typer.Option(
        4,
        "--refine-iters",
        help="Local refinement iterations per top candidate.",
    ),
    refine_step_px: float = typer.Option(
        180.0,
        "--refine-step-px",
        help="Initial local refinement step in pixels.",
    ),
    nodata_min: Optional[float] = typer.Option(
        None,
        "--nodata-min",
        help="Treat sampled slope values >= this threshold as invalid (optional).",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for repeatability."),
    show_top: int = typer.Option(10, "--show-top", help="How many ranked candidates to print."),
    self_test: bool = typer.Option(
        False,
        "--self-test",
        help="Run a synthetic TERCOM test by sampling a random known location.",
    ),
    self_test_points: int = typer.Option(
        120,
        "--self-test-points",
        help="Number of points for synthetic profile generation.",
    ),
    self_test_noise: float = typer.Option(
        3.0,
        "--self-test-noise",
        help="Gaussian noise sigma added in self-test mode.",
    ),
) -> None:
    terrain_map = TerrainMap(map_path)

    try:
        if self_test:
            code = run_self_test(
                terrain_map=terrain_map,
                spacing_m=spacing_m,
                heading_deg=heading_deg,
                headings=headings,
                samples=samples,
                topk=topk,
                refine_iters=refine_iters,
                refine_step_px=refine_step_px,
                nodata_min=nodata_min,
                seed=seed,
                self_test_points=self_test_points,
                self_test_noise=self_test_noise,
            )
            raise typer.Exit(code=code)

        if profile is None and stock_symbol is None:
            raise typer.BadParameter(
                "Provide exactly one input source: either --profile or --stock-symbol."
            )
        if profile is not None and stock_symbol is not None:
            raise typer.BadParameter("Use only one source: --profile or --stock-symbol, not both.")

        if stock_symbol is not None:
            api_key = load_alpha_vantage_api_key(alpha_env_file)
            profile_data = fetch_stock_profile(
                symbol=stock_symbol.upper(),
                points=stock_points,
                field=stock_field,
                api_key=api_key,
            )
            typer.echo(
                f"Loaded {profile_data.size} historical '{stock_field}' points for {stock_symbol.upper()} from AlphaVantage."
            )
        else:
            profile_data = load_profile(profile)

        if heading_deg is not None:
            heading_values = np.asarray([heading_deg], dtype=np.float64)
        else:
            heading_values = np.linspace(0.0, 360.0, headings, endpoint=False)

        rng = np.random.default_rng(seed)
        results = search_tercom(
            terrain_map=terrain_map,
            profile=profile_data,
            spacing_m=spacing_m,
            headings_deg=heading_values,
            random_samples=samples,
            top_k=topk,
            refine_iters=refine_iters,
            refine_step_px=refine_step_px,
            nodata_min=nodata_min,
            rng=rng,
        )

        if not results:
            typer.echo("No valid candidate found.")
            raise typer.Exit(code=1)

        count_to_print = 10 if stock_symbol is not None else show_top
        typer.echo("Top similar elevations on the map")
        for idx, match in enumerate(results[:count_to_print], start=1):
            typer.echo(format_match(idx, match, terrain_map))
    finally:
        terrain_map.close()


if __name__ == "__main__":
    app()
