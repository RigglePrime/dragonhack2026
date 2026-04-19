from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from gemini_query import choose_coordinate
from logging_config import setup_logging
from terrain_estimator import CoordinateCandidate, estimate_candidates

setup_logging()
logger = logging.getLogger(__name__)

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
    path_mode: Literal["straight", "astar"] = "straight"
    headings: int = 12
    random_samples: int = 500
    top_k: int = 10
    refine_iters: int = 2
    refine_step_px: float = 120.0


app = FastAPI(title="Terrain Vibe Service", version="1.0.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    logger.info(
        "request.start id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.exception("request.error id=%s elapsed_ms=%.2f", request_id, elapsed_ms)
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request.end id=%s status=%s elapsed_ms=%.2f",
        request_id,
        response.status_code,
        elapsed_ms,
    )
    return response


def resolve_symbol(query: str) -> str:
    """
    Accepts either a ticker (AAPL) or a company name (Apple).
    Returns the best matching ticker symbol.
    """
    import yfinance as yf

    query = query.strip()
    if not query:
        logger.warning("resolve_symbol.empty_query")
        raise ValueError("Empty symbol or company name")

    # If user already typed a ticker, try it directly
    try:
        test = yf.Ticker(query).history(period="1d")
        if not test.empty:
            logger.info("resolve_symbol.direct_success query=%s symbol=%s", query, query.upper())
            return query.upper()
    except Exception:
        logger.exception("resolve_symbol.direct_error query=%s", query)

    # Otherwise search by company name
    try:
        search_result = yf.Search(query=query, max_results=1)
        quotes = getattr(search_result, "quotes", [])
        if quotes and "symbol" in quotes[0]:
            logger.info(
                "resolve_symbol.search_success query=%s symbol=%s",
                query,
                quotes[0]["symbol"].upper(),
            )
            return quotes[0]["symbol"].upper()
    except Exception:
        logger.exception("resolve_symbol.search_error query=%s", query)

    logger.warning("resolve_symbol.failed query=%s", query)
    raise ValueError(f"Could not resolve '{query}' to a stock ticker")

def _compute_bounds(points: list[dict[str, float]]) -> dict:
    if not points:
        logger.warning("bounds.empty_points_using_default")
        return {
            "min_lat": 49.0,
            "max_lat": 51.0,
            "min_lon": 9.0,
            "max_lon": 11.0,
        }
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
    logger.info("stock.fetch.start symbol=%s window=%s", symbol, window)
    config = WINDOW_CONFIG[window]
    history = yf.Ticker(symbol).history(
        period=config["period"],
        interval=config["interval"],
        auto_adjust=False,
    )
    if history.empty:
        logger.warning("stock.fetch.empty symbol=%s window=%s", symbol, window)
        raise ValueError(f"No Yahoo Finance data found for symbol '{symbol}'.")

    closes = history["Close"].dropna()
    if closes.empty:
        logger.warning("stock.fetch.no_close symbol=%s window=%s", symbol, window)
        raise ValueError(f"No close-price data found for symbol '{symbol}'.")

    series: list[dict[str, float | str]] = []
    for timestamp, close in closes.items():
        ts_any: Any = timestamp
        try:
            if getattr(ts_any, "tzinfo", None) is not None and hasattr(ts_any, "tz_convert"):
                ts_any = ts_any.tz_convert("UTC")
            if hasattr(ts_any, "tz_localize"):
                ts_any = ts_any.tz_localize(None)
        except Exception:
            pass
        time_text = ts_any.isoformat() if hasattr(ts_any, "isoformat") else str(ts_any)
        series.append({"time": time_text, "close": float(close)})
    logger.info("stock.fetch.done symbol=%s points=%d", symbol, len(series))
    return series


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    logger.info(
        "analyze.start symbol_input=%s window=%s path_mode=%s spacing_m=%.2f headings=%d",
        request.symbol,
        request.window,
        request.path_mode,
        request.spacing_m,
        request.headings,
    )
    symbol = resolve_symbol(request.symbol)

    if not symbol:
        raise HTTPException(status_code=400, detail="symbol must not be empty")

    try:
        series = _fetch_stock_series(symbol=symbol, window=request.window)
    except Exception as exc:
        logger.exception("analyze.stock_fetch_failed symbol=%s", symbol)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    closes = np.asarray([point["close"] for point in series], dtype=np.float64)
    map_path = Path(os.getenv("TERRAIN_MAP_PATH", DEFAULT_MAP_PATH))
    if not map_path.exists():
        logger.error("analyze.map_missing path=%s", map_path)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Terrain map not found at '{map_path}'. "
                "Set TERRAIN_MAP_PATH or mount the map into the container."
            ),
        )

    try:
        terrain_start = time.perf_counter()
        candidates = estimate_candidates(
            profile=closes,
            map_path=map_path,
            top_n=10,
            spacing_m=request.spacing_m,
            path_mode=request.path_mode,
            headings=request.headings,
            random_samples=request.random_samples,
            top_k=request.top_k,
            refine_iters=request.refine_iters,
            refine_step_px=request.refine_step_px,
            seed=42,
        )
        logger.info(
            "analyze.terrain_done symbol=%s candidates=%d elapsed_ms=%.2f",
            symbol,
            len(candidates),
            (time.perf_counter() - terrain_start) * 1000.0,
        )
    except Exception as exc:
        logger.exception("analyze.terrain_failed symbol=%s", symbol)
        raise HTTPException(status_code=500, detail=f"Terrain estimation failed: {exc}") from exc

    if not candidates:
        raise HTTPException(status_code=500, detail="Terrain estimation returned no candidates.")

    candidate_payload = [_candidate_to_dict(candidate) for candidate in candidates]

    company = (request.company or symbol).strip() or symbol
    gemini_start = time.perf_counter()
    gemini = choose_coordinate(company=company, places=candidate_payload)
    logger.info(
        "analyze.gemini_done chosen_rank=%d fallback=%s elapsed_ms=%.2f",
        gemini.chosen_rank,
        gemini.used_fallback,
        (time.perf_counter() - gemini_start) * 1000.0,
    )

    selected_rank = gemini.chosen_rank
    selected = next((c for c in candidate_payload if c["rank"] == selected_rank), candidate_payload[0])
    # Zoomed-out: all candidate points
    zoomed_out_bounds=_compute_bounds(candidate_payload)

    # Zoomed-in: only the selected route
    zoomed_in_bounds=_compute_bounds(selected.get("route", []))

    logger.info(
        "analyze.done symbol=%s selected_rank=%d route_points=%d",
        symbol,
        selected_rank,
        len(selected.get("route", [])),
    )

    return {
        "symbol": symbol,
        "window": request.window,
        "path_mode": request.path_mode,
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
