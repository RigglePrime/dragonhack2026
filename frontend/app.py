from __future__ import annotations

import logging
import os
import time

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Take a Hike!", layout="wide")
st.title("Take a Hike!")
st.caption("Yahoo Finance -> Terrain Matching -> Gemini Pick -> OpenStreetMap")


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "trigger_search" not in st.session_state:
    st.session_state.trigger_search = False

def trigger_search():
    logger.debug("frontend.trigger_search")
    st.session_state.trigger_search = True

def _render_zoomed_out_map(zoomed_out: dict, selected_route: list[dict], selected_rank: int) -> folium.Map:
    bounds = zoomed_out["bounds"]
    points = zoomed_out["points"]

    center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
    center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="OpenStreetMap")

    # --- Draw all points except chosen (red) and first (yellow) ---
    for p in points:
        if p["rank"] in (selected_rank, 1):
            continue
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=5,
            color="blue",
            fill=True,
            fill_opacity=0.85,
            popup=f"Rank {p['rank']}",
        ).add_to(fmap)

    # --- Rank #1 in yellow ---
    first = next((p for p in points if p["rank"] == 1), None)
    if first:
        folium.CircleMarker(
            location=[first["lat"], first["lon"]],
            radius=8,
            color="yellow",
            fill=True,
            fill_opacity=1.0,
            popup="Rank 1 (best score)",
        ).add_to(fmap)

    # --- Chosen candidate in red ---
    chosen = next((p for p in points if p["rank"] == selected_rank), None)
    if chosen:
        folium.CircleMarker(
            location=[chosen["lat"], chosen["lon"]],
            radius=10,
            color="red",
            fill=True,
            fill_opacity=1.0,
            popup=f"Rank {selected_rank} (chosen)",
        ).add_to(fmap)

    # --- Chosen route in red ---
    if selected_route:
        folium.PolyLine(
            locations=[(p["lat"], p["lon"]) for p in selected_route],
            color="red",
            weight=3,
            opacity=0.8,
        ).add_to(fmap)

    # --- Legend ---
    legend_html="""
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        font-size: 12px;
        color: black;
        background-color: rgba(255, 255, 255, 0.0);  /* fully transparent */
        padding: 4px 6px;
        line-height: 1.2;
    ">
        <b style="color:black;">Legend</b><br>
        <span style="color: blue;">●</span> Other<br>
        <span style="color: yellow;">●</span> Rank #1<br>
        <span style="color: red;">●</span> Chosen<br>
    </div>
    """
    root = fmap.get_root()
    html_root = getattr(root, "html", None)
    if html_root is not None:
        html_root.add_child(folium.Element(legend_html))

    fmap.fit_bounds([
        [bounds["min_lat"], bounds["min_lon"]],
        [bounds["max_lat"], bounds["max_lon"]],
    ])

    return fmap



def _render_zoomed_in_map(zoomed_in: dict) -> folium.Map:
    bounds = zoomed_in["bounds"]
    route = zoomed_in["route"]

    # Add padding (5% of span)
    lat_pad = (bounds["max_lat"] - bounds["min_lat"]) * 0.05
    lon_pad = (bounds["max_lon"] - bounds["min_lon"]) * 0.05

    padded_bounds = [
        [bounds["min_lat"] - lat_pad, bounds["min_lon"] - lon_pad],
        [bounds["max_lat"] + lat_pad, bounds["max_lon"] + lon_pad],
    ]

    center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
    center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    # Draw route
    if route:
        folium.PolyLine(
            locations=[(p["lat"], p["lon"]) for p in route],
            color="red",
            weight=4,
            opacity=0.85,
        ).add_to(fmap)

        folium.Marker([route[0]["lat"], route[0]["lon"]], popup="Start").add_to(fmap)
        folium.Marker([route[-1]["lat"], route[-1]["lon"]], popup="End").add_to(fmap)

    # Fit padded bounds
    fmap.fit_bounds(padded_bounds)

    return fmap

def _render_map(candidates: list[dict], selected_candidate: dict | None, route: list[dict]) -> folium.Map:
    if selected_candidate is not None:
        center = [selected_candidate["lat"], selected_candidate["lon"]]
        zoom = 5
    elif candidates:
        center = [candidates[0]["lat"], candidates[0]["lon"]]
        zoom = 4
    else:
        center = [50.0, 10.0]
        zoom = 4

    fmap = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

    selected_rank = selected_candidate.get("rank") if selected_candidate else None
    for candidate in candidates:
        is_selected = candidate["rank"] == selected_rank
        color = "red" if is_selected else "blue"
        popup = (
            f"Rank #{candidate['rank']}<br>"
            f"Score: {candidate['score']:.4f}<br>"
            f"Lat: {candidate['lat']:.6f}, Lon: {candidate['lon']:.6f}<br>"
            f"Heading: {candidate['heading_deg']:.2f}"
        )
        folium.CircleMarker(
            location=[candidate["lat"], candidate["lon"]],
            radius=7 if is_selected else 5,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=popup,
        ).add_to(fmap)

    if route and len(route) >= 2:
        folium.PolyLine(
            locations=[(point["lat"], point["lon"]) for point in route],
            color="red",
            weight=4,
            opacity=0.85,
        ).add_to(fmap)

    return fmap


left, right = st.columns([2, 1])
with left:
    symbol=st.text_input("Stock market identifier", placeholder="AAPL or Tesla or Apple", key="symbol_input", on_change=trigger_search)

with right:
    window = st.selectbox(
        "History window",
        options=["1d", "1w", "1mo"],
        index=2,
        help="Choose whether to use last day, week, or month of price history.",
    )

path_mode = st.selectbox(
    "Path mode",
    options=["straight", "astar"],
    index=0,
    help="Use A* for non-straight route estimation or straight-line baseline.",
)

company_override = st.text_input(
    "Company vibe label (optional)",
    placeholder="If empty, symbol is used in the Gemini prompt.",
)

with st.expander("Advanced matching options"):
    spacing_m = st.number_input("Spacing meters", min_value=1.0, value=100.0, step=1.0)
    headings = st.slider("Heading hypotheses", min_value=4, max_value=36, value=12)
    random_samples = st.slider("Random samples", min_value=50, max_value=2000, value=500, step=50)
    top_k = st.slider("Top-k coarse candidates", min_value=10, max_value=100, value=10, step=5)
    refine_iters = st.slider("Refine iterations", min_value=1, max_value=6, value=2)

run_now = st.button("Analyze") or st.session_state.trigger_search

if run_now:
    st.session_state.trigger_search = False  # reset flag

    symbol_clean = symbol.strip()
    if not symbol_clean:
        logger.warning("frontend.run.missing_symbol")
        st.error("Please enter a stock identifier or company name.")
    else:
        payload = {
            "symbol": symbol_clean,
            "window": window,
            "path_mode": path_mode,
            "spacing_m": spacing_m,
            "headings": headings,
            "random_samples": random_samples,
            "top_k": top_k,
            "refine_iters": refine_iters,
        }
        if company_override.strip():
            payload["company"] = company_override.strip()

        try:
            logger.info(
                "frontend.analyze.start symbol=%s window=%s path_mode=%s",
                symbol_clean,
                window,
                path_mode,
            )
            started = time.perf_counter()
            with st.spinner("Fetching stock data, estimating terrain, asking Gemini..."):
                response = requests.post(
                    f"{BACKEND_URL}/api/analyze",
                    json=payload,
                    timeout=300,
                )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "frontend.analyze.response status=%d elapsed_ms=%.1f",
                response.status_code,
                elapsed_ms,
            )
            if response.status_code != 200:
                logger.error("frontend.analyze.error status=%d body=%s", response.status_code, response.text)
                st.error(f"Backend error {response.status_code}: {response.text}")
            else:
                st.session_state.analysis_result = response.json()
                logger.info("frontend.analyze.success")
        except Exception as exc:
            logger.exception("frontend.analyze.exception")
            st.error(f"Request failed: {exc}")


result = st.session_state.analysis_result
if result:

    logger.debug(
        "frontend.render.result symbol=%s candidates=%d",
        result.get("symbol", ""),
        len(result.get("candidates", [])),
    )



    gemini = result.get("gemini", {})
    st.markdown("### Gemini pick")
    st.write(f"Chosen rank: {gemini.get('chosen_rank', 'n/a')}")
    st.write(gemini.get("reason", "No explanation available."))

    candidates = result.get("candidates", [])
    selected = result.get("selected_candidate")
    route = result.get("route", [])
    terrain_profile=result.get("terrain_profile", [])
    # Min–max normalize terrain profile
    terrain_norm=None
    if terrain_profile:
        terrain_arr=pd.Series(terrain_profile, dtype=float)
        terrain_norm=(terrain_arr-terrain_arr.min())/(terrain_arr.max()-terrain_arr.min())

    st.markdown("### Maps")

    maps=result.get("maps", {})

    col1, col2=st.columns(2)

    with col1:
        st.markdown("#### 🌍 Zoomed‑out (all candidates + highlighted route)")
        if "zoomed_out" in maps:
            selected_rank=selected["rank"]
            fmap_out=_render_zoomed_out_map(maps["zoomed_out"], route, selected_rank)

            st_folium(fmap_out, width=600, height=500)
        else:
            st.info("Zoomed‑out map not available.")

    with col2:
        st.markdown("#### 🗺️ Zoomed‑in (selected route, slightly zoomed out)")
        if "zoomed_in" in maps:
            fmap_in=_render_zoomed_in_map(maps["zoomed_in"])
            st_folium(fmap_in, width=600, height=500)
        else:
            st.info("Zoomed‑in map not available.")
    st.subheader(
        f"Symbol: {result['symbol']} ({result['window']}) [path: {result.get('path_mode', 'n/a')}]"
    )
    series_df=pd.DataFrame(result["series"])
    if not series_df.empty:
        series_df["time"] = pd.to_datetime(series_df["time"])
        series_df = series_df.sort_values("time")
        df = series_df.set_index("time")

        close = df["close"]

        # Min–max normalization
        normalized = (close - close.min()) / (close.max() - close.min())

        st.line_chart(normalized, use_container_width=True)
        #st.line_chart(series_df.set_index("time")["close"], use_container_width=True)
    import altair as alt

    if candidates and terrain_profile:
        st.markdown("### Terrain Height Profile (Chosen vs Rank #1)")

        chosen_rank=selected["rank"]
        first_rank=1

        # --- Extract profiles ---
        chosen_profile=None
        first_profile=None

        for c in candidates:
            if c["rank"]==chosen_rank:
                chosen_profile=c.get("terrain_profile", [])
            if c["rank"]==first_rank:
                first_profile=c.get("terrain_profile", [])

        if chosen_profile and first_profile:
            # --- Min–max normalize both ---
            chosen_norm=(pd.Series(chosen_profile)-min(chosen_profile))/(max(chosen_profile)-min(chosen_profile))
            first_norm=(pd.Series(first_profile)-min(first_profile))/(max(first_profile)-min(first_profile))

            # --- Build dataframe ---
            df=pd.DataFrame({"index":range(len(chosen_norm)), "chosen":chosen_norm, "first":first_norm, })

            # --- Melt for Altair ---
            df_melt=df.melt("index", var_name="series", value_name="value")

            # --- Color mapping ---
            color_scale=alt.Scale(domain=["chosen", "first"], range=["red", "yellow"])

            chart=(alt.Chart(df_melt).mark_line(strokeWidth=3).encode(x=alt.X("index:Q", title="Sample Index"), y=alt.Y("value:Q", title="Normalized Slope"), color=alt.Color("series:N", scale=color_scale, title="Legend"), tooltip=["series", "index", "value"], ).properties(width=900, height=400))

            st.altair_chart(chart, use_container_width=True)

    if candidates:
        st.markdown("### All Terrain Profiles (Chosen in Red, Rank #1 in Yellow)")

        chosen_rank=selected["rank"]
        first_rank=1

        rows=[]
        for c in candidates:
            profile=c.get("terrain_profile", [])
            for i, v in enumerate(profile):
                rows.append({"index":i, "value":v, "rank":str(c["rank"]), "color_group":("chosen" if c["rank"]==chosen_rank else "first" if c["rank"]==first_rank else "other")})

        df_all=pd.DataFrame(rows)

        # Color mapping
        color_scale=alt.Scale(domain=["chosen", "first", "other"], range=["red", "yellow", "blue"])

        chart=(alt.Chart(df_all).mark_line().encode(x=alt.X("index:Q", title="Sample Index"), y=alt.Y("value:Q", title="Slope Degrees"), color=alt.Color("color_group:N", scale=color_scale, title="Legend"), detail="rank:N", tooltip=["rank", "index", "value"], ).properties(width=900, height=400))

        st.altair_chart(chart, use_container_width=True)

    if candidates:
        st.markdown("### Top 10 candidates")
        table_df = pd.DataFrame(candidates)
        keep_cols = ["rank", "score", "lat", "lon", "heading_deg"]
        st.dataframe(table_df[keep_cols], use_container_width=True)

