from __future__ import annotations

import os

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Stock Terrain Vibes", layout="wide")
st.title("Stock Terrain Vibes")
st.caption("Yahoo Finance -> Terrain Matching -> Gemini Pick -> OpenStreetMap")


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


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
    symbol = st.text_input("Stock market identifier", placeholder="AAPL")
with right:
    window = st.selectbox(
        "History window",
        options=["1d", "1w", "1mo"],
        index=2,
        help="Choose whether to use last day, week, or month of price history.",
    )

company_override = st.text_input(
    "Company vibe label (optional)",
    placeholder="If empty, symbol is used in the Gemini prompt.",
)

with st.expander("Advanced matching options"):
    spacing_m = st.number_input("Spacing meters", min_value=1.0, value=25.0, step=1.0)
    headings = st.slider("Heading hypotheses", min_value=4, max_value=36, value=12)
    random_samples = st.slider("Random samples", min_value=50, max_value=2000, value=300, step=50)
    top_k = st.slider("Top-k coarse candidates", min_value=10, max_value=100, value=20, step=5)
    refine_iters = st.slider("Refine iterations", min_value=1, max_value=6, value=2)

if st.button("Analyze"):
    symbol_clean = symbol.strip().upper()
    if not symbol_clean:
        st.error("Please enter a stock identifier.")
    else:
        payload = {
            "symbol": symbol_clean,
            "window": window,
            "spacing_m": spacing_m,
            "headings": headings,
            "random_samples": random_samples,
            "top_k": top_k,
            "refine_iters": refine_iters,
        }
        if company_override.strip():
            payload["company"] = company_override.strip()

        try:
            with st.spinner("Fetching stock data, estimating terrain, asking Gemini..."):
                response = requests.post(
                    f"{BACKEND_URL}/api/analyze",
                    json=payload,
                    timeout=300,
                )
            if response.status_code != 200:
                st.error(f"Backend error {response.status_code}: {response.text}")
            else:
                st.session_state.analysis_result = response.json()
        except Exception as exc:
            st.error(f"Request failed: {exc}")

result = st.session_state.analysis_result
if result:
    st.subheader(f"Symbol: {result['symbol']} ({result['window']})")
    series_df = pd.DataFrame(result["series"])
    if not series_df.empty:
        series_df["time"] = pd.to_datetime(series_df["time"])
        series_df = series_df.sort_values("time")
        st.line_chart(series_df.set_index("time")["close"], use_container_width=True)

    gemini = result.get("gemini", {})
    st.markdown("### Gemini pick")
    st.write(f"Chosen rank: {gemini.get('chosen_rank', 'n/a')}")
    st.write(gemini.get("reason", "No explanation available."))

    candidates = result.get("candidates", [])
    selected = result.get("selected_candidate")
    route = result.get("route", [])

    st.markdown("### Map")
    fmap = _render_map(candidates=candidates, selected_candidate=selected, route=route)
    st_folium(fmap, width=1200, height=520)

    if candidates:
        st.markdown("### Top 10 candidates")
        table_df = pd.DataFrame(candidates)
        keep_cols = ["rank", "score", "lat", "lon", "heading_deg"]
        st.dataframe(table_df[keep_cols], use_container_width=True)
