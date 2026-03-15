from pathlib import Path

import folium
import geopandas as gpd
import streamlit as st
import google.generativeai as genai
from shapely.geometry import Point
from streamlit_folium import st_folium


st.set_page_config(
    page_title="Water Quality Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

BASEMAPS = {
    "Light": {"tiles": "CartoDB positron", "attr": "CartoDB"},
    "Street": {"tiles": "OpenStreetMap", "attr": "OpenStreetMap"},
    "Satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri",
    },
}

CATEGORY_ORDER = ["Acidic", "Safe", "Alkaline"]


def load_css() -> None:
    css_path = BASE_DIR / "style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def classify_ph(ph_value: float) -> str:
    if ph_value < 6.5:
        return "Acidic"
    if ph_value <= 8.5:
        return "Safe"
    return "Alkaline"


def marker_color(ph_value: float) -> str:
    if ph_value < 6.5:
        return "#ef4444"
    if ph_value <= 8.5:
        return "#22c55e"
    return "#2563eb"


def metric_block(title: str, value: str, note: str) -> str:
    return (
        "<div class='metric-shell'>"
        f"<p class='metric-kicker'>{title}</p>"
        f"<p class='metric-number'>{value}</p>"
        f"<p class='metric-note'>{note}</p>"
        "</div>"
    )


@st.cache_data(show_spinner=False)
def load_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    boundary = gpd.read_file(DATA_DIR / "hisar_boundary.geojson").to_crs("EPSG:4326")
    stations = gpd.read_file(DATA_DIR / "hisar_water_validated.geojson").to_crs("EPSG:4326")

    location_candidates = ["Location", "location", "Village", "village", "Name", "name"]
    location_col = next((col for col in location_candidates if col in stations.columns), None)
    ph_col = next((col for col in stations.columns if col.lower() == "ph"), None)

    if location_col is None:
        raise ValueError("A location column was not found in your dataset.")
    if ph_col is None:
        raise ValueError("A pH column was not found in your dataset.")

    stations = stations.rename(columns={location_col: "location", ph_col: "ph"}).copy()
    stations["ph"] = stations["ph"].astype(float)
    stations = stations.dropna(subset=["geometry", "ph"]).copy()
    stations["lat"] = stations.geometry.y
    stations["lon"] = stations.geometry.x
    stations["category"] = stations["ph"].apply(classify_ph)
    stations["marker_color"] = stations["ph"].apply(marker_color)

    return boundary, stations


def tooltip_html(row) -> str:
    return (
        "<div style='min-width:190px;padding:10px 12px;border-radius:12px;"
        "background:linear-gradient(155deg,#0f172a,#1e293b);"
        "color:#e2e8f0;border:1px solid #38bdf8;box-shadow:0 10px 24px rgba(2,6,23,0.45);"
        "font-family:Outfit,Segoe UI,sans-serif;'>"
        f"<div style='font-size:13px;font-weight:700;color:#f8fafc;margin-bottom:6px;'>{row['location']}</div>"
        f"<div style='display:flex;justify-content:space-between;gap:14px;font-size:12px;'><span>pH</span><strong>{row['ph']:.2f}</strong></div>"
        f"<div style='display:flex;justify-content:space-between;gap:14px;font-size:12px;'><span>Status</span><strong>{row['category']}</strong></div>"
        f"<div style='display:flex;justify-content:space-between;gap:14px;font-size:12px;'><span>Lat</span><strong>{row['lat']:.4f}</strong></div>"
        f"<div style='display:flex;justify-content:space-between;gap:14px;font-size:12px;'><span>Lon</span><strong>{row['lon']:.4f}</strong></div>"
        "</div>"
    )


def build_map(
    boundary: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    center_lat: float,
    center_lon: float,
    basemap: str,
) -> folium.Map:
    map_obj = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(**BASEMAPS[basemap], name=basemap).add_to(map_obj)

    folium.GeoJson(
        boundary,
        name="District Boundary",
        style_function=lambda _: {
            "fillColor": "#67e8f9",
            "fillOpacity": 0.08,
            "color": "#0f172a",
            "weight": 2.3,
        },
        highlight_function=lambda _: {
            "fillColor": "#22d3ee",
            "fillOpacity": 0.14,
            "color": "#020617",
            "weight": 3.2,
        },
    ).add_to(map_obj)

    map_hover_css = """
    <style>
    path.leaflet-interactive {
        transition: all .18s ease;
    }
    path.leaflet-interactive:hover {
        stroke: #ffffff !important;
        stroke-width: 4 !important;
        fill-opacity: 1 !important;
    }
    </style>
    """
    map_obj.get_root().html.add_child(folium.Element(map_hover_css))

    for _, row in stations.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color="#0f172a",
            weight=1.4,
            fill=True,
            fill_color=row["marker_color"],
            fill_opacity=0.88,
            tooltip=folium.Tooltip(tooltip_html(row), sticky=True, direction="top"),
        ).add_to(map_obj)

    legend_html = """
    <div class="map-legend">
        <div class="legend-title">pH Range</div>
        <div><span class="legend-dot acidic"></span>Acidic (&lt; 6.5)</div>
        <div><span class="legend-dot safe"></span>Safe (6.5 - 8.5)</div>
        <div><span class="legend-dot alkaline"></span>Alkaline (&gt; 8.5)</div>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=True).add_to(map_obj)
    return map_obj


def nearest_station(stations: gpd.GeoDataFrame, lat: float, lon: float):
    if stations.empty:
        return None
    click_point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    stations_proj = stations.to_crs("EPSG:3857")
    nearest_idx = stations_proj.geometry.distance(click_point).idxmin()
    return stations.loc[nearest_idx]


def build_chat_context(filtered_gdf):
    if filtered_gdf.empty:
        return "No stations currently match the filters."
    total = len(filtered_gdf)
    acidic = (filtered_gdf["category"] == "Acidic").sum()
    safe = (filtered_gdf["category"] == "Safe").sum()
    alkaline = (filtered_gdf["category"] == "Alkaline").sum()
    avg_ph = filtered_gdf["ph"].mean()
    min_ph = filtered_gdf["ph"].min()
    max_ph = filtered_gdf["ph"].max()
    top_locations = filtered_gdf.nlargest(3, "ph")[["location", "ph"]].to_dict("records")
    low_locations = filtered_gdf.nsmallest(3, "ph")[["location", "ph"]].to_dict("records")

    context = f"""
You are an assistant for a water quality dashboard in Hisar district. 
Current filter stats:
- Total stations: {total}
- Acidic (<6.5): {acidic}
- Safe (6.5–8.5): {safe}
- Alkaline (>8.5): {alkaline}
- pH range: {min_ph:.2f} – {max_ph:.2f}
- Average pH: {avg_ph:.2f}

Highest pH stations: {top_locations}
Lowest pH stations: {low_locations}

Answer the user's question concisely and helpfully, using this data if relevant.
"""
    return context


load_css()
boundary_gdf, station_gdf = load_data()

# ---------- GEMINI CONFIG & SESSION INIT ----------
GEMINI_API_KEY = "AIzaSyD0gMqxHxljnQzHRyJ3vTouoNgNgEQG4gg"
genai.configure(api_key=GEMINI_API_KEY)
import google.generativeai as genai

genai.configure(api_key="AIzaSyD0gMqxHxljnQzHRyJ3vTouoNgNgEQG4gg")
models = genai.list_models()
for model in models:
    if 'generateContent' in model.supported_generation_methods:
        print(model.name)
model = genai.GenerativeModel("gemini-3-flash-preview")

if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = []

st.markdown(
    """
    <section class="hero">
        <p class="hero-label">Groundwater Monitoring Platform</p>
        <h1 class="hero-title">Water Quality Intelligence Dashboard</h1>
        <p class="hero-subtitle">
            Live geospatial monitoring of station-level pH readings with hover-enabled location cards,
            district boundary context, and decision-friendly analytics.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Control Panel")
    selected_basemap = st.selectbox("Map Theme", list(BASEMAPS.keys()), index=0)

    min_ph = float(station_gdf["ph"].min())
    max_ph = float(station_gdf["ph"].max())
    ph_range = st.slider(
        "pH Filter",
        min_value=min_ph,
        max_value=max_ph,
        value=(min_ph, max_ph),
        step=0.01,
    )

    categories = st.multiselect(
        "Quality Class",
        options=CATEGORY_ORDER,
        default=CATEGORY_ORDER,
    )

    search_station = st.text_input("Find Station")

    # ---- COMPUTE filtered HERE (so it's available for the chat) ----
    filtered = station_gdf[
        (station_gdf["ph"] >= ph_range[0])
        & (station_gdf["ph"] <= ph_range[1])
        & (station_gdf["category"].isin(categories))
    ].copy()

    if search_station.strip():
        filtered = filtered[
            filtered["location"].str.contains(search_station.strip(), case=False, na=False)
        ].copy()

    # ---- AI ASSISTANT (now filtered is defined) ----
    st.divider()
    with st.expander("💬 AI Assistant", expanded=False):
        st.caption("Ask about water quality")
        # Display chat history
        for msg in st.session_state.gemini_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask Gemini..."):
            # Add user message
            st.session_state.gemini_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build context using filtered (defined above)
            context = build_chat_context(filtered)
            full_prompt = f"{context}\n\nUser question: {prompt}"
            try:
                response = model.generate_content(full_prompt)
                reply = response.text
            except Exception as e:
                reply = f"Error calling Gemini: {e}"

            # Add assistant message
            st.session_state.gemini_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

filtered = station_gdf[
    (station_gdf["ph"] >= ph_range[0])
    & (station_gdf["ph"] <= ph_range[1])
    & (station_gdf["category"].isin(categories))
].copy()

if search_station.strip():
    filtered = filtered[
        filtered["location"].str.contains(search_station.strip(), case=False, na=False)
    ].copy()

active_count = int(len(filtered))
safe_count = int((filtered["category"] == "Safe").sum())
risk_count = int((filtered["category"] != "Safe").sum())
avg_ph = float(filtered["ph"].mean()) if not filtered.empty else 0.0

metric_cols = st.columns(4)
with metric_cols[0]:
    st.markdown(metric_block("Active Stations", f"{active_count}", "Filtered monitoring points"), unsafe_allow_html=True)
with metric_cols[1]:
    st.markdown(metric_block("Average pH", f"{avg_ph:.2f}", "Current district snapshot"), unsafe_allow_html=True)
with metric_cols[2]:
    st.markdown(metric_block("Safe Stations", f"{safe_count}", "Within WHO-like safe bracket"), unsafe_allow_html=True)
with metric_cols[3]:
    st.markdown(metric_block("Attention Needed", f"{risk_count}", "Acidic or alkaline locations"), unsafe_allow_html=True)

if filtered.empty:
    map_center_lat = float(station_gdf["lat"].mean())
    map_center_lon = float(station_gdf["lon"].mean())
else:
    map_center_lat = float(filtered["lat"].mean())
    map_center_lon = float(filtered["lon"].mean())

map_col, insight_col = st.columns([3.2, 1.35], gap="large")

with map_col:
    st.markdown("<h3 class='panel-heading'>Live Water Quality Map</h3>", unsafe_allow_html=True)
    live_map = build_map(
        boundary=boundary_gdf,
        stations=filtered,
        center_lat=map_center_lat,
        center_lon=map_center_lon,
        basemap=selected_basemap,
    )
    try:
        map_state = st_folium(live_map, use_container_width=True, height=700)
    except TypeError:
        map_state = st_folium(live_map, width=1200, height=700)

with insight_col:
    st.markdown("<h3 class='panel-heading'>Station Insights</h3>", unsafe_allow_html=True)

    selected_station = None
    if map_state and map_state.get("last_clicked"):
        selected_station = nearest_station(
            filtered,
            map_state["last_clicked"]["lat"],
            map_state["last_clicked"]["lng"],
        )

    if selected_station is None and not filtered.empty:
        selected_station = filtered.loc[(filtered["ph"] - 7.0).abs().idxmax()]

    if selected_station is not None:
        st.markdown(
            f"""
            <div class="insight-card">
                <p class="insight-label">Focused Station</p>
                <h4>{selected_station['location']}</h4>
                <p><strong>pH:</strong> {selected_station['ph']:.2f}</p>
                <p><strong>Category:</strong> {selected_station['category']}</p>
                <p><strong>Latitude:</strong> {selected_station['lat']:.6f}</p>
                <p><strong>Longitude:</strong> {selected_station['lon']:.6f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(selected_station["ph"] / 14.0, 0), 1))
    else:
        st.markdown(
            "<div class='empty-state'>No station matches current filters. Expand pH range or categories.</div>",
            unsafe_allow_html=True,
        )

    category_counts = (
        filtered["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
        if not filtered.empty
        else {"Acidic": 0, "Safe": 0, "Alkaline": 0}
    )

    st.markdown(
        f"""
        <div class="insight-card">
            <p class="insight-label">Filter Summary</p>
            <p><strong>Acidic:</strong> {int(category_counts['Acidic'])}</p>
            <p><strong>Safe:</strong> {int(category_counts['Safe'])}</p>
            <p><strong>Alkaline:</strong> {int(category_counts['Alkaline'])}</p>
            <p><strong>Average pH:</strong> {avg_ph:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

analysis_col_1, analysis_col_2 = st.columns([1.25, 2.35], gap="large")

with analysis_col_1:
    st.markdown("<h3 class='panel-heading'>Quality Distribution</h3>", unsafe_allow_html=True)
    distribution = (
        filtered["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
        if not filtered.empty
        else None
    )
    if distribution is None:
        st.info("No data to visualize for current filters.")
    else:
        st.bar_chart(distribution)

with analysis_col_2:
    st.markdown("<h3 class='panel-heading'>Station Table</h3>", unsafe_allow_html=True)
    if filtered.empty:
        st.warning("No records available for the selected filters.")
    else:
        table = (
            filtered[["location", "ph", "category", "lat", "lon"]]
            .sort_values(by="ph", ascending=False)
            .rename(
                columns={
                    "location": "Location",
                    "ph": "pH",
                    "category": "Category",
                    "lat": "Latitude",
                    "lon": "Longitude",
                }
            )
        )
        st.dataframe(table, use_container_width=True, height=360)
  