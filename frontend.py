import streamlit as st
import yfinance as yf
import requests
import folium
from streamlit_folium import st_folium

st.title("Stock Finder + Map")

# --- SESSION STATE SETUP ---
if "symbol" not in st.session_state:
    st.session_state.symbol = None

if "data" not in st.session_state:
    st.session_state.data = None

if "map" not in st.session_state:
    st.session_state.map = None


# --- SEARCH FUNCTION ---
def search_ticker(query):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 1, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params, headers=headers)

    try:
        data = response.json()
    except:
        return None

    if "quotes" in data and len(data["quotes"]) > 0:
        return data["quotes"][0]["symbol"]
    return None


# --- UI INPUT ---
query = st.text_input("Enter company name:")

if st.button("Search"):
    symbol = search_ticker(query)

    if symbol:
        st.session_state.symbol = symbol
        st.session_state.data = yf.Ticker(symbol).history(period="1mo")

        # Create map ONCE and store it
        m = folium.Map(location=[46.0569, 14.5058], zoom_start=12)
        folium.Marker([46.0569, 14.5058], popup="Ljubljana").add_to(m)
        st.session_state.map = m
    else:
        st.error("No matching stock found.")


# --- DISPLAY RESULTS (PERSISTENT) ---
if st.session_state.symbol:
    st.success(f"Found symbol: {st.session_state.symbol}")
    scaled = (st.session_state.data["Close"] - st.session_state.data["Close"].min()) / (st.session_state.data["Close"].max() - st.session_state.data["Close"].min())
    st.line_chart(scaled)

if st.session_state.map:
    st.subheader("OpenStreetMap")
    st_folium(st.session_state.map, width=700, height=500)