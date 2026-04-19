from __future__ import annotations

import heapq
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import rasterio
import typer
import yfinance as yf
from rasterio.windows import Window

VALID_DN_MIN = 0.0
VALID_DN_MAX = 250.0
PATH_MODE = Literal["straight", "astar"]

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

    def rowscols_to_latlon(self, rows: np.ndarray, cols: np.ndarray) -> list[tuple[float, float]]:
        route: list[tuple[float, float]] = []
        for row, col in zip(rows, cols):
            x, y = self.pixel_to_projected(float(row), float(col))
            latlon = self.projected_to_wgs84(x, y)
            if latlon is not None:
                route.append(latlon)
        return route

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

    def sample_points(self, rows: np.ndarray, cols: np.ndarray) -> Optional[np.ndarray]:
        if rows.size == 0 or cols.size == 0 or rows.size != cols.size:
            return None
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
        return self.sample_points(rows, cols)


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


def _minmax_scale_to_slope(profile: np.ndarray) -> np.ndarray:
    pmin = float(np.min(profile))
    pmax = float(np.max(profile))
    if pmax - pmin < 1e-9:
        return np.full_like(profile, 45.0, dtype=np.float64)
    scaled = (profile - pmin) / (pmax - pmin)
    return 90.0 * scaled


def _resample_polyline(
    rows: np.ndarray,
    cols: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if rows.size < 2 or cols.size < 2 or n_samples < 2:
        return rows, cols

    seg_lengths = np.sqrt(np.diff(rows) ** 2 + np.diff(cols) ** 2)
    cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-9:
        return rows[:n_samples], cols[:n_samples]

    targets = np.linspace(0.0, total_length, n_samples)
    out_rows = np.interp(targets, cumulative, rows)
    out_cols = np.interp(targets, cumulative, cols)
    return out_rows, out_cols


def _astar_guided_route(
    terrain_map: TerrainMap,
    target_scaled: np.ndarray,
    start_row: float,
    start_col: float,
    goal_row: float,
    goal_col: float,
    corridor_pad: int = 32,
    max_iters: int = 80000,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    top = max(1, int(min(start_row, goal_row)) - corridor_pad)
    bottom = min(terrain_map.height - 2, int(max(start_row, goal_row)) + corridor_pad)
    left = max(1, int(min(start_col, goal_col)) - corridor_pad)
    right = min(terrain_map.width - 2, int(max(start_col, goal_col)) + corridor_pad)

    if bottom <= top + 2 or right <= left + 2:
        return None

    row_idx = np.arange(top, bottom + 1, dtype=np.float64)
    col_idx = np.arange(left, right + 1, dtype=np.float64)
    grid_rows, grid_cols = np.meshgrid(row_idx, col_idx, indexing="ij")
    slope_patch = terrain_map.sample_points(grid_rows.ravel(), grid_cols.ravel())
    if slope_patch is None:
        return None
    slope_patch = slope_patch.reshape((row_idx.size, col_idx.size))

    def to_local(r: float, c: float) -> tuple[int, int]:
        return int(round(r)) - top, int(round(c)) - left

    sr, sc = to_local(start_row, start_col)
    gr, gc = to_local(goal_row, goal_col)

    if sr < 0 or sc < 0 or gr < 0 or gc < 0:
        return None
    if sr >= slope_patch.shape[0] or sc >= slope_patch.shape[1]:
        return None
    if gr >= slope_patch.shape[0] or gc >= slope_patch.shape[1]:
        return None

    neighbor_moves = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    expected_steps = max(1.0, math.hypot(gr - sr, gc - sc))
    max_steps = int(expected_steps * 2.5) + 8

    open_heap: list[tuple[float, float, int, int, int]] = []
    start_state = (sr, sc, 0)
    g_cost: dict[tuple[int, int, int], float] = {start_state: 0.0}
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    start_h = math.hypot(gr - sr, gc - sc)
    heapq.heappush(open_heap, (start_h, 0.0, sr, sc, 0))

    goal_state: Optional[tuple[int, int, int]] = None
    iterations = 0

    while open_heap and iterations < max_iters:
        iterations += 1
        f, g, r, c, steps = heapq.heappop(open_heap)
        current = (r, c, steps)

        if g > g_cost.get(current, float("inf")):
            continue

        if r == gr and c == gc:
            goal_state = current
            break

        if steps >= max_steps:
            continue

        for dr, dc in neighbor_moves:
            nr = r + dr
            nc = c + dc
            if nr <= 0 or nc <= 0 or nr >= slope_patch.shape[0] - 1 or nc >= slope_patch.shape[1] - 1:
                continue

            next_steps = steps + 1
            progress = min(1.0, next_steps / max(1.0, expected_steps))
            target_idx = int(progress * (target_scaled.size - 1))
            expected_slope = target_scaled[target_idx]
            local_slope = float(slope_patch[nr, nc])

            mismatch_penalty = abs(local_slope - expected_slope) / 90.0
            move_cost = math.hypot(dr, dc)
            g_new = g + move_cost + 1.5 * mismatch_penalty

            state = (nr, nc, next_steps)
            if g_new >= g_cost.get(state, float("inf")):
                continue

            g_cost[state] = g_new
            parent[state] = current
            h = 0.6 * math.hypot(gr - nr, gc - nc)
            heapq.heappush(open_heap, (g_new + h, g_new, nr, nc, next_steps))

    if goal_state is None:
        return None

    path_local: list[tuple[int, int]] = []
    s = goal_state
    while True:
        path_local.append((s[0], s[1]))
        if s == start_state:
            break
        s = parent[s]

    path_local.reverse()
    path_rows = np.asarray([top + r for r, _ in path_local], dtype=np.float64)
    path_cols = np.asarray([left + c for _, c in path_local], dtype=np.float64)
    return path_rows, path_cols


def _candidate_path_profile(
    terrain_map: TerrainMap,
    profile: np.ndarray,
    row: float,
    col: float,
    heading_deg: float,
    spacing_px: float,
    path_mode: PATH_MODE,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if path_mode == "straight":
        line = terrain_map.sample_line(row, col, heading_deg, spacing_px, profile.size)
        if line is None:
            return None
        route_rows, route_cols = terrain_map._line_offsets(row, col, heading_deg, spacing_px, profile.size)
        return line, route_rows, route_cols

    endpoints_rows, endpoints_cols = terrain_map._line_offsets(
        row,
        col,
        heading_deg,
        spacing_px,
        profile.size,
    )
    if not terrain_map._validate_bounds(endpoints_rows, endpoints_cols):
        return None

    target_scaled = _minmax_scale_to_slope(profile)
    route = _astar_guided_route(
        terrain_map=terrain_map,
        target_scaled=target_scaled,
        start_row=float(endpoints_rows[0]),
        start_col=float(endpoints_cols[0]),
        goal_row=float(endpoints_rows[-1]),
        goal_col=float(endpoints_cols[-1]),
    )
    if route is None:
        return None

    route_rows, route_cols = route
    sample_rows, sample_cols = _resample_polyline(route_rows, route_cols, n_samples=profile.size)
    terrain_profile = terrain_map.sample_points(sample_rows, sample_cols)
    if terrain_profile is None:
        return None

    return terrain_profile, sample_rows, sample_cols


def evaluate_candidate(
    terrain_map: TerrainMap,
    profile: np.ndarray,
    row: float,
    col: float,
    heading_deg: float,
    spacing_px: float,
    path_mode: PATH_MODE,
) -> tuple[float, float]:
    sampled = _candidate_path_profile(
        terrain_map=terrain_map,
        profile=profile,
        row=row,
        col=col,
        heading_deg=heading_deg,
        spacing_px=spacing_px,
        path_mode=path_mode,
    )
    if sampled is None:
        return float("inf"), -2.0

    terrain_profile, _, _ = sampled
    distance = profile_distance(profile, terrain_profile)
    if not np.isfinite(distance):
        return float("inf"), -2.0

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
    path_mode: PATH_MODE,
) -> list[MatchResult]:
    _ = refine_iters, refine_step_px  # Kept for API compatibility.

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

    heading_values = headings_deg

    # Two-stage strategy for non-straight mode:
    # 1) evaluate all 100 placed locations with fast straight mode,
    # 2) run A* only on the best shortlist.
    eval_mode: PATH_MODE = "straight" if path_mode == "astar" else path_mode

    for _ in range(placed_locations):
        row = float(rng.integers(row_low, row_high))
        col = float(rng.integers(col_low, col_high))

        best_local: Optional[MatchResult] = None
        for heading in heading_values:
            distance, score = evaluate_candidate(
                terrain_map=terrain_map,
                profile=profile,
                row=row,
                col=col,
                heading_deg=float(heading),
                spacing_px=spacing_px,
                path_mode=eval_mode,
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

    if path_mode != "astar":
        return closest

    shortlist = candidates[: min(24, len(candidates))]
    refined: list[MatchResult] = []
    for candidate in shortlist:
        distance, score = evaluate_candidate(
            terrain_map=terrain_map,
            profile=profile,
            row=candidate.row,
            col=candidate.col,
            heading_deg=candidate.heading_deg,
            spacing_px=spacing_px,
            path_mode="astar",
        )
        if np.isfinite(distance):
            refined.append(
                MatchResult(
                    distance=distance,
                    score=score,
                    row=candidate.row,
                    col=candidate.col,
                    heading_deg=candidate.heading_deg,
                )
            )

    if not refined:
        return closest

    refined.sort(key=lambda item: item.distance)
    return refined[:selected_count]


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
    path_mode: PATH_MODE = "straight",
) -> list[CoordinateCandidate]:
    _ = random_samples, top_k  # Preserved for request/CLI compatibility.

    clean_profile = np.asarray(profile, dtype=np.float64)
    clean_profile = clean_profile[np.isfinite(clean_profile)]
    if clean_profile.size < 8:
        raise ValueError("Profile must contain at least 8 valid points.")

    headings_deg = np.linspace(0.0, 360.0, headings, endpoint=False)
    rng = np.random.default_rng(seed)

    # Requested behavior: place 100 locations and keep the 10 closest.
    placed_locations = 100
    selected_count = 10 if top_n > 0 else 0

    terrain_map = TerrainMap(map_path)
    try:
        matches = search_tercom(
            terrain_map=terrain_map,
            profile=clean_profile,
            spacing_m=spacing_m,
            headings_deg=headings_deg,
            random_samples=placed_locations,
            top_k=10,
            refine_iters=refine_iters,
            refine_step_px=refine_step_px,
            rng=rng,
            path_mode=path_mode,
        )

        spacing_px = spacing_m / terrain_map.scale_x
        candidates: list[CoordinateCandidate] = []

        for rank, match in enumerate(matches[:selected_count], start=1):
            sampled = _candidate_path_profile(
                terrain_map=terrain_map,
                profile=clean_profile,
                row=match.row,
                col=match.col,
                heading_deg=match.heading_deg,
                spacing_px=spacing_px,
                path_mode=path_mode,
            )
            if sampled is None and path_mode == "astar":
                sampled = _candidate_path_profile(
                    terrain_map=terrain_map,
                    profile=clean_profile,
                    row=match.row,
                    col=match.col,
                    heading_deg=match.heading_deg,
                    spacing_px=spacing_px,
                    path_mode="straight",
                )
            if sampled is None:
                continue

            terrain_profile, sample_rows, sample_cols = sampled
            x, y = terrain_map.pixel_to_projected(match.row, match.col)
            latlon = terrain_map.projected_to_wgs84(x, y)
            if latlon is None:
                continue

            lat, lon = latlon
            route = terrain_map.rowscols_to_latlon(sample_rows, sample_cols)

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
                    terrain_profile=terrain_profile.tolist(),
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
    random_samples: int = typer.Option(100, "--samples"),
    top_k: int = typer.Option(10, "--top-k"),
    refine_iters: int = typer.Option(2, "--refine-iters"),
    refine_step_px: float = typer.Option(120.0, "--refine-step-px"),
    seed: int = typer.Option(42, "--seed"),
    path_mode: PATH_MODE = typer.Option("straight", "--path-mode"),
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
        path_mode=path_mode,
    )

    if as_json:
        payload: list[dict] = []
        for candidate in candidates:
            item = asdict(candidate)
            item["route_latlon"] = [{"lat": lat, "lon": lon} for lat, lon in candidate.route_latlon]
            payload.append(item)
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Top {len(candidates)} candidates for {symbol.upper()} ({window}) [{path_mode}]")
    for candidate in candidates:
        typer.echo(
            f"#{candidate.rank} score={candidate.score:.4f} lat={candidate.lat:.6f} "
            f"lon={candidate.lon:.6f} heading={candidate.heading_deg:.2f}"
        )


if __name__ == "__main__":
    app()
