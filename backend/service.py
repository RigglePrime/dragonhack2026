from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from gemini_query import choose_coordinate
from terrain_estimator import CoordinateCandidate, estimate_candidates

WINDOW_CONFIG = {
    "1d": {"period": "1d", "interval": "5m"},
    "1w": {"period": "5d", "interval": "30m"},
    "1mo": {"period": "1mo", "interval": "1d"},
}

DEFAULT_MAP_PATH = "eud_cp_slop/eudem_slop_3035_europe.tif"


class AnalyzeRequest(BaseModel):
    symbol: str = Field(min_length=1, description="Ticker symbol, e.g. AAPL")
    window: Literal["1d", "1w", "1mo"] = "1mo"
    company: Optional[str] = Field(
        default=None,
        description="Company name used in Gemini prompt. Defaults to symbol.",
    )
    spacing_m: float = 100.0
    headings: int = 12
    random_samples: int = 500
    top_k: int = 10
    refine_iters: int = 2
    refine_step_px: float = 120.0


app = FastAPI(title="Terrain Vibe Service", version="1.0.0")


def resolve_symbol(query: str) -> str:
    """
    Accepts either a ticker (AAPL) or a company name (Apple).
    Returns the best matching ticker symbol.
    """
    import yfinance as yf

    query = query.strip()
    if not query:
        raise ValueError("Empty symbol or company name")

    # If user already typed a ticker, try it directly
    try:
        test = yf.Ticker(query).history(period="1d")
        if not test.empty:
            return query.upper()
    except Exception:
        pass

    # Otherwise search by company name
    try:
        results = yf.search(query)
        if results and "symbol" in results[0]:
            return results[0]["symbol"].upper()
    except Exception:
        pass

    raise ValueError(f"Could not resolve '{query}' to a stock ticker")

def _compute_bounds(points: list[dict[str, float]]) -> dict:
    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    return {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons),
    }
def _candidate_to_dict(candidate: CoordinateCandidate) -> dict:
    return {
        "rank": candidate.rank,
        "score": candidate.score,
        "row": candidate.row,
        "col": candidate.col,
        "heading_deg": candidate.heading_deg,
        "projected_x": candidate.projected_x,
        "projected_y": candidate.projected_y,
        "lat": candidate.lat,
        "lon": candidate.lon,
        "route": [{"lat": lat, "lon": lon} for lat, lon in candidate.route_latlon],
        "terrain_profile":candidate.terrain_profile

    }


def _fetch_stock_series(symbol: str, window: str) -> list[dict[str, float | str]]:
    config = WINDOW_CONFIG[window]
    history = yf.Ticker(symbol).history(
        period=config["period"],
        interval=config["interval"],
        auto_adjust=False,
    )
    if history.empty:
        raise ValueError(f"No Yahoo Finance data found for symbol '{symbol}'.")

    closes = history["Close"].dropna()
    if closes.empty:
        raise ValueError(f"No close-price data found for symbol '{symbol}'.")

    series: list[dict[str, float | str]] = []
    for timestamp, close in closes.items():
        ts = timestamp
        try:
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
        except Exception:
            pass
        series.append({"time": ts.isoformat(), "close": float(close)})
    return series


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    symbol = resolve_symbol(request.symbol)

    if not symbol:
        raise HTTPException(status_code=400, detail="symbol must not be empty")

    try:
        series = _fetch_stock_series(symbol=symbol, window=request.window)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    closes = np.asarray([point["close"] for point in series], dtype=np.float64)
    map_path = Path(os.getenv("TERRAIN_MAP_PATH", DEFAULT_MAP_PATH))
    if not map_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Terrain map not found at '{map_path}'. "
                "Set TERRAIN_MAP_PATH or mount the map into the container."
            ),
        )

    try:
        candidates = estimate_candidates(
            profile=closes,
            map_path=map_path,
            top_n=10,
            spacing_m=request.spacing_m,
            headings=request.headings,
            random_samples=request.random_samples,
            top_k=request.top_k,
            refine_iters=request.refine_iters,
            refine_step_px=request.refine_step_px,
            seed=42,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Terrain estimation failed: {exc}") from exc

    if not candidates:
        raise HTTPException(status_code=500, detail="Terrain estimation returned no candidates.")

    candidate_payload = [_candidate_to_dict(candidate) for candidate in candidates]

    company = (request.company or symbol).strip() or symbol
    gemini = choose_coordinate(company=company, places=candidate_payload)

    selected_rank = gemini.chosen_rank
    selected = next((c for c in candidate_payload if c["rank"] == selected_rank), candidate_payload[0])
    # --- MAP BOUNDS ---
    # NEW: include terrain profile for selected candidate
    selected_profile=selected.get("terrain_profile", [])

    # Zoomed-out: all candidate points
    zoomed_out_bounds=_compute_bounds(candidate_payload)

    # Zoomed-in: only the selected route
    zoomed_in_bounds=_compute_bounds(selected.get("route", []))

    return {
        "symbol": symbol,
        "window": request.window,
        "company": company,
        "series": series,
        "candidates": candidate_payload,
        "gemini": {
            "chosen_rank": gemini.chosen_rank,
            "reason": gemini.reason,
            "raw_text": gemini.raw_text,
            "used_fallback": gemini.used_fallback,
        },
        "selected_candidate": selected,
        "route": selected.get("route", []),
        "maps":{"zoomed_out":{"bounds":zoomed_out_bounds, "points":[{"lat":c["lat"], "lon":c["lon"], "rank":c["rank"]} for c in candidate_payload], },
                "zoomed_in":{"bounds":zoomed_in_bounds, "route":selected.get("route", []), }, },
        "terrain_profile":selected.get("terrain_profile", []),

    }
