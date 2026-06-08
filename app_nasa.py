import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from sklearn.linear_model import LinearRegression
import calendar

# -------------------------
# Configuration
# -------------------------
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="🌾 Climate Impact & Food Security Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stMetricLabel"] { font-size: 0.8rem; }
[data-testid="stMetricValue"] { font-size: 1.4rem; }
[data-testid="stMetricDelta"] { font-size: 0.75rem; }
[data-testid="metric-container"] { padding: 10px; }
.main-header {
    font-size: 2.5rem; font-weight: 700; color: #2E7D32;
    margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 3px solid #4CAF50;
}
.sub-header {
    font-size: 1.8rem; font-weight: 600; color: #388E3C;
    margin-top: 1.5rem; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">Climate Impact on Malaysian Food Availability Dashboard</div>',
    unsafe_allow_html=True
)

# -------------------------
# State coordinates & Normalisation Map
# -------------------------
STATE_COORDS = {
    "Johor":             (1.4927,  103.7414),
    "Kedah":             (6.1254,  100.3678),
    "Kelantan":          (6.1252,  102.2382),
    "Melaka":            (2.1896,  102.2501),
    "Negeri Sembilan":   (2.7254,  101.9420),
    "Pahang":            (3.8079,  102.5485),
    "Penang":            (5.4164,  100.3327),
    "Pulau Pinang":      (5.4164,  100.3327),
    "Perak":             (4.5975,  101.0901),
    "Perlis":            (6.4425,  100.2083),
    "Sabah":             (5.9804,  116.0735),
    "Sarawak":           (1.5533,  110.3593),
    "Selangor":          (3.0738,  101.5183),
    "Terengganu":        (5.3333,  103.1333),
    "Kuala Lumpur":      (3.1390,  101.6869),
    "W.P. Kuala Lumpur": (3.1390,  101.6869),
    "Labuan":            (5.2830,  115.2340),
    "W.P. Labuan":       (5.2830,  115.2340),
    "Putrajaya":         (2.9264,  101.6963),
    "Malaysia":          (4.2105,  101.9758),
}

NASA_STATE_MAP = {
    "Pulau Pinang": "Penang",
}

# -------------------------
# Helper functions
# -------------------------
def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
              .str.replace("\u00A0", "", regex=False)
              .str.replace(",",      "", regex=False)
              .str.replace(r"[^0-9.\-]", "", regex=True)
              .replace("", np.nan),
        errors="coerce",
    )


def _normalise_state(name: str) -> str:
    name = str(name).strip().title()
    return NASA_STATE_MAP.get(name, name)


# -------------------------
# NASA climate loader
# -------------------------
@st.cache_data
def load_nasa_climate() -> pd.DataFrame | None:
    possible_paths = [
        "climate_state_2017_2022.csv",
        "./climate_state_2017_2022.csv",
        "../climate_state_2017_2022.csv",
        "./data/climate_state_2017_2022.csv",
        "/Users/izzulfidaey/PycharmProjects/FYP/climate_state_2017_2022.csv",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            climate = pd.read_csv(path)
            climate["state"] = climate["state"].apply(_normalise_state)
            climate["year"] = climate["year"].astype(int)
            return climate

    uploaded = st.sidebar.file_uploader(
        "Upload climate_state_2017_2022.csv (NASA POWER)",
        type="csv",
        key="nasa_climate_uploader",
    )
    if uploaded is not None:
        climate = pd.read_csv(uploaded)
        climate["state"] = climate["state"].apply(_normalise_state)
        climate["year"] = climate["year"].astype(int)
        st.sidebar.success("✅ NASA climate data loaded!")
        return climate

    return None


# -------------------------
# Main data loader
# -------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    # ── 1. Load crops CSV ───────────────────────────────────────────────
    possible_paths = [
        "/Users/izzulfidaey/PycharmProjects/FYP/crops_state.csv",
        "/Users/izzulfidaey/Desktop/FYP/crops_state.csv",
        "crops_state.csv",
        "./crops_state.csv",
        "../crops_state.csv",
        "./data/crops_state.csv",
    ]

    file_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if file_path is None:
        st.sidebar.warning("crops_state.csv not found.")
        uploaded_file = st.sidebar.file_uploader(
            "Upload crops_state.csv", type="csv", key="crops_uploader"
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding="latin1")
            st.sidebar.success("Crops file uploaded!")
        else:
            st.sidebar.info("Using sample data for demonstration.")
            return create_sample_data()
    else:
        df = pd.read_csv(file_path, encoding="latin1")

    # ── 2. Clean column names & state names ─────────────────────────────
    df.columns = [c.strip() for c in df.columns]
    
    if "state" in df.columns:
        df["state"] = df["state"].apply(_normalise_state)

    # ── 3. Clean crop types ─────────────────────────────────────────────
    if "crop_type" in df.columns:
        df["crop_type"] = df["crop_type"].astype(str).str.strip().str.title()
        crop_mapping = {
            "Cash_Crops":      "Corn", "Industrial_Crops": "Palm Oil",
            "cash_crops":      "Corn", "industrial_crops": "Palm Oil",
            "Cash Crops":      "Corn", "Industrial Crops": "Palm Oil",
        }
        df["crop_type"] = df["crop_type"].replace(crop_mapping)

    # ── 4. Clean production, area, & original weather units ─────────────
    for col in ["temperature", "humidity", "production", "planted_area"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace("°C", "", regex=False)
                       .str.replace("%", "",  regex=False)
                       .str.replace("。", "", regex=False)
                       .str.replace("吽C", "", regex=False)
            )
            df[col] = clean_numeric(df[col])

    # ── 5. Parse dates → year & month ───────────────────────────────────
    if "date" in df.columns:
        df["date"]  = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
        if df["date"].isna().all():
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"]  = df["date"].dt.year.fillna(2017).astype(int)
        df["month"] = df["date"].dt.month.fillna(1).astype(int)

    if "year" not in df.columns:
        df["year"] = 2017

    # ── 6. Merge NASA POWER climate data ────────────────────────────────
    nasa = load_nasa_climate()

    if nasa is not None:
        nasa = nasa.rename(columns={
            "mean_temp_c":       "temperature",
            "mean_humidity_pct": "humidity",
        })

        # Safeguard: Drop existing climate metrics to merge fresh NASA API data
        df = df.drop(columns=[c for c in ["temperature", "humidity"] if c in df.columns])

        df = df.merge(
            nasa[["state", "year", "temperature", "humidity"]],
            on=["state", "year"],
            how="left",
        )

        matched   = df["temperature"].notna().sum()
        unmatched = df["temperature"].isna().sum()

        if unmatched == 0:
            st.sidebar.success(f"✅ NASA climate merged: all {matched} rows covered")
        else:
            st.sidebar.warning(
                f"⚠️ NASA climate: {matched}/{len(df)} rows matched. "
                f"{unmatched} rows missing climate data."
            )
    else:
        st.sidebar.warning(
            "⚠️ NASA climate CSV not found — falling back to original metrics from crops CSV. "
            "Run app_nasa.py to generate it."
        )

    # ── 7. Derived metrics ───────────────────────────────────────────────
    if {"production", "planted_area"}.issubset(df.columns):
        df["yield_efficiency"] = np.where(
            df["planted_area"] > 0,
            df["production"] / df["planted_area"],
            np.nan,
        )

    # ── 8. Coordinates ───────────────────────────────────────────────────
    df["lat"] = df["state"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[0])
    df["lon"] = df["state"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[1])

    return df


# -------------------------
# Sample data fallback
# -------------------------
def create_sample_data() -> pd.DataFrame:
    states = ["Johor", "Kedah", "Kelantan", "Melaka", "Selangor"]
    crops  = ["Corn", "Palm Oil", "Vegetables", "Fruits", "Paddy"]
    years  = [2017, 2018, 2019, 2020, 2021, 2022]
    data   = []
    for state in states:
        for crop in crops:
            for year in years:
                data.append({
                    "state":        state,
                    "crop_type":    crop,
                    "year":         year,
                    "production":   np.random.randint(10000, 100000) * (1 + 0.05 * (year - 2017)),
                    "planted_area": np.random.randint(1000, 10000),
                    "temperature":  np.random.uniform(25, 32) + (year - 2017) * 0.1,
                    "humidity":     np.random.uniform(70, 90),
                    "lat":          4.0 + np.random.uniform(-1, 1),
                    "lon":          102.0 + np.random.uniform(-2, 2),
                })
    df = pd.DataFrame(data)
    df["yield_efficiency"] = df["production"] / df["planted_area"]
    return df


# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()

if df.empty:
    st.error("No data available. Please check your data file.")
    st.stop()

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Dashboard Controls")

tab_selection = st.sidebar.radio(
    "Select Dashboard View:",
    ["📊 Summary & Overview", "🔍 Trend Analysis", "🎯 Climate Simulation"]
)

state_options = sorted([s for s in df["state"].dropna().unique() if s.lower() != "malaysia"])
crop_options  = sorted(df["crop_type"].dropna().unique())

states_selected = st.sidebar.multiselect("Select State(s):",  options=state_options, default=state_options)
crops_selected  = st.sidebar.multiselect("Select Crop Type(s):", options=crop_options,  default=crop_options)

min_year = int(df["year"].min()) if df["year"].notnull().any() else 2017
max_year = int(df["year"].max()) if df["year"].notnull().any() else 2022

# Fix: Prevent Streamlit API Exception when there's only 1 year available
if min_year == max_year:
    st.sidebar.markdown(f"**Year Range:** {min_year}")
    year_range = (min_year, max_year)
else:
    year_range = st.sidebar.slider("Year Range:", min_year, max_year, (min_year, max_year))

filtered = df[
    (df["state"].isin(states_selected)) &
    (df["crop_type"].isin(crops_selected)) &
    (df["year"].between(year_range[0], year_range[1]))
].copy()

if filtered.empty:
    st.warning("No data after applying filters. Please adjust your selections.")
    st.stop()

# Additional computed columns
filtered["climate_stress_index"] = (
    (filtered["temperature"] - filtered["temperature"].mean()) +
    (filtered["humidity"]    - filtered["humidity"].mean())
)

filtered = filtered.sort_values(["state", "crop_type", "year"])
filtered["yoy_growth"]          = filtered.groupby(["state", "crop_type"])["production"].pct_change() * 100
filtered["yoy_temp_change"]     = filtered.groupby(["state", "crop_type"])["temperature"].diff()
filtered["yoy_humidity_change"] = filtered.groupby(["state", "crop_type"])["humidity"].diff()

st.sidebar.info(f"Showing {len(filtered)} records")

# =============================================================================
# TAB 1 — SUMMARY & OVERVIEW
# =============================================================================
if tab_selection == "📊 Summary & Overview":
    st.markdown('<div class="sub-header">National Overview & Key Insights</div>', unsafe_allow_html=True)

    available_years = sorted(filtered["year"].unique())
    if len(available_years) >= 2:
        latest_year = available_years[-1]
        prev_year   = available_years[-2]
        latest_data = filtered[filtered["year"] == latest_year]
        prev_data   = filtered[filtered["year"] == prev_year]
        prod_delta  = ((latest_data["production"].sum()        - prev_data["production"].sum())        / prev_data["production"].sum())        * 100 if prev_data["production"].sum()        != 0 else 0
        yield_delta = ((latest_data["yield_efficiency"].mean() - prev_data["yield_efficiency"].mean()) / prev_data["yield_efficiency"].mean()) * 100 if prev_data["yield_efficiency"].mean() != 0 else 0
        temp_delta  = latest_data["temperature"].mean() - prev_data["temperature"].mean()
    else:
        prod_delta = yield_delta = temp_delta = 0

    # KPI row 1
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Production",   f"{filtered['production'].sum():,.0f} MT",     f"{prod_delta:+.1f}% YoY")
    with kpi2:
        st.metric("Avg Yield Efficiency", f"{filtered['yield_efficiency'].mean():.2f} MT/Ha", f"{yield_delta:+.1f}% YoY")
    with kpi3:
        st.metric("Avg Temperature (NASA)", f"{filtered['temperature'].mean():.1f}°C",   f"{temp_delta:+.1f}°C", delta_color="inverse")
    with kpi4:
        st.metric("Total Planted Area", f"{filtered['planted_area'].sum():,.0f} Ha",    "Cumulative")

    # KPI row 2
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Avg Humidity (NASA)", f"{filtered['humidity'].mean():.1f}%")
    with col6:
        st.metric("States Covered",  str(filtered["state"].nunique()))
    with col7:
        st.metric("Crop Types",      str(filtered["crop_type"].nunique()))
    with col8:
        st.metric("Years of Data",   str(filtered["year"].nunique()))

    # NASA data source note
    st.info(
        "🛰️ **Temperature & Humidity source:** NASA POWER API  "
        "(T2M = 2-metre air temperature, RH2M = 2-metre relative humidity) — "
        "annual means aggregated from daily values per state capital coordinates."
    )

    # Top performing states
    st.markdown('<div class="sub-header">Top Performing States</div>', unsafe_allow_html=True)

    state_performance = filtered.groupby("state").agg(
        production=("production",      "sum"),
        yield_efficiency=("yield_efficiency", "mean"),
        temperature=("temperature",    "mean"),
        humidity=("humidity",          "mean"),
    ).reset_index()

    st.markdown("**Top 5 States by Production**")
    top_production = state_performance.nlargest(5, "production")
    prod_cols = st.columns(5)
    for col, (_, row) in zip(prod_cols, top_production.iterrows()):
        with col:
            st.markdown(f"""
            <div style="background:#e8f5e9;padding:12px;border-radius:8px;text-align:center;border-left:4px solid #4CAF50;">
                <div style="font-weight:bold;color:#1a472a;">{row['state']}</div>
                <div style="font-size:1.1rem;font-weight:bold;">{row['production']:,.0f} MT</div>
                <div style="font-size:0.8rem;color:#555;">{row['temperature']:.1f}°C | {row['humidity']:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)

    st.markdown("**Top 5 States by Yield Efficiency**")
    top_yield = state_performance.nlargest(5, "yield_efficiency")
    yield_cols = st.columns(5)
    for col, (_, row) in zip(yield_cols, top_yield.iterrows()):
        with col:
            st.markdown(f"""
            <div style="background:#e3f2fd;padding:12px;border-radius:8px;text-align:center;border-left:4px solid #2196F3;">
                <div style="font-weight:bold;color:#0d47a1;">{row['state']}</div>
                <div style="font-size:1.1rem;font-weight:bold;">{row['yield_efficiency']:.2f} MT/Ha</div>
                <div style="font-size:0.8rem;color:#555;">{row['temperature']:.1f}°C | {row['humidity']:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    # Production map
    st.markdown('<div class="sub-header">Production Distribution Map</div>', unsafe_allow_html=True)

    def clean_state_name(state):
        if isinstance(state, str):
            state = state.strip()
            lower = state.lower()
            if "pulau pinang" in lower or "penang" in lower: return "Pulau Pinang"
            if "kuala lumpur" in lower:                       return "W.P. Kuala Lumpur"
            if "labuan" in lower:                             return "W.P. Labuan"
            if "putrajaya" in lower:                          return "Putrajaya"
            return state
        return state

    filtered_clean             = filtered.copy()
    filtered_clean["state_clean"] = filtered_clean["state"].apply(clean_state_name)

    state_stats = filtered_clean.groupby("state_clean", as_index=False).agg(
        total_production=("production",  "sum"),
        total_planted_area=("planted_area", "sum"),
        avg_temp=("temperature", "mean"),
        avg_humidity=("humidity", "mean"),
    ).rename(columns={"state_clean": "state"})

    state_stats["avg_yield"] = (state_stats["total_production"] / state_stats["total_planted_area"]).round(2)
    state_stats = state_stats.replace([np.inf, -np.inf], 0).fillna(0)

    try:
        highest_crop = filtered_clean.loc[
            filtered_clean.groupby("state_clean")["production"].idxmax()
        ].set_index("state_clean")[["crop_type"]].rename(columns={"crop_type": "highest_crop"})
        lowest_crop  = filtered_clean.loc[
            filtered_clean.groupby("state_clean")["production"].idxmin()
        ].set_index("state_clean")[["crop_type"]].rename(columns={"crop_type": "lowest_crop"})
    except Exception:
        highest_crop = lowest_crop = pd.DataFrame()

    map_df = state_stats.set_index("state").join(highest_crop, how="left").join(lowest_crop, how="left").reset_index()
    map_df["lat"] = map_df["state"].map(lambda x: STATE_COORDS.get(x, (np.nan, np.nan))[0])
    map_df["lon"] = map_df["state"].map(lambda x: STATE_COORDS.get(x, (np.nan, np.nan))[1])
    map_df = map_df.dropna(subset=["lat", "lon"])

    if not map_df.empty:
        min_prod = map_df["total_production"].min()
        max_prod = map_df["total_production"].max()
        if max_prod > min_prod:
            map_df["production_norm"] = (map_df["total_production"] - min_prod) / (max_prod - min_prod)
            map_df["radius"]          = map_df["production_norm"] * 50000 + 10000
        else:
            map_df["production_norm"] = 0.5
            map_df["radius"]          = 20000

        map_df["color"]        = map_df["production_norm"].apply(lambda v: [0, int(100 + v * 155), 0, 180])
        map_df["highest_crop"] = map_df.get("highest_crop", pd.Series()).fillna("No data")
        map_df["lowest_crop"]  = map_df.get("lowest_crop",  pd.Series()).fillna("No data")

        bubble_layer = pdk.Layer(
            "ScatterplotLayer", data=map_df,
            get_position="[lon, lat]", get_radius="radius",
            radius_min_pixels=10, radius_max_pixels=60,
            get_fill_color="color", pickable=True, stroked=True,
            line_width_min_pixels=1, get_line_color=[0, 80, 0, 200],
            auto_highlight=True,
        )
        text_layer = pdk.Layer(
            "TextLayer", data=map_df,
            get_position="[lon, lat]", get_text="state",
            get_color=[0, 0, 0, 255], get_size=60,
            get_alignment_baseline="'center'", get_pixel_offset=[0, -15],
            font_weight="bold",
        )
        tooltip = {
            "html": """
            <div style="font-family:Arial;background:white;border-radius:8px;
                        box-shadow:0 4px 15px rgba(0,0,0,0.2);padding:12px;
                        border:2px solid #4CAF50;color:#333;min-width:230px;">
                <div style="border-bottom:2px solid #eee;padding-bottom:8px;margin-bottom:8px;
                            font-size:16px;font-weight:800;text-align:center;color:#1a472a;">
                    {state}
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span>Production:</span><span style="font-weight:bold;">{total_production} MT</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span>Planted Area:</span><span style="font-weight:bold;">{total_planted_area} Ha</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span>Avg Yield:</span><span style="font-weight:bold;color:#2E7D32;">{avg_yield} MT/Ha</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span>🌡 Avg Temp (NASA):</span><span style="font-weight:bold;">{avg_temp}°C</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                    <span>💧 Avg Humidity (NASA):</span><span style="font-weight:bold;">{avg_humidity}%</span>
                </div>
                <div style="border-top:1px dashed #ccc;margin:5px 0 10px 0;"></div>
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <div style="background:#e8f5e9;color:#2e7d32;padding:2px 6px;border-radius:4px;
                                font-size:11px;font-weight:bold;">HIGHEST CROP</div>
                    <span style="font-weight:600;">{highest_crop}</span>
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <div style="background:#ffebee;color:#c62828;padding:2px 6px;border-radius:4px;
                                font-size:11px;font-weight:bold;">LOWEST CROP</div>
                    <span style="font-weight:600;">{lowest_crop}</span>
                </div>
            </div>""",
            "style": {"backgroundColor": "transparent"},
        }
        view_state = pdk.ViewState(
            latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(),
            zoom=5.5, pitch=0,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[bubble_layer, text_layer],
            initial_view_state=view_state,
            tooltip=tooltip, map_style="light", height=550,
        ))

# =============================================================================
# TAB 2 — TREND ANALYSIS
# =============================================================================
elif tab_selection == "🔍 Trend Analysis":

    st.markdown('<div class="sub-header">Crop Distribution Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        sel_state = st.selectbox("Select State:", filtered["state"].unique(), key="dist_state")
        state_df  = filtered[filtered["state"] == sel_state]

        if not state_df.empty:
            st.markdown(f"""
            <div style="background:#1a472a;padding:15px;border-radius:10px;color:white;margin-bottom:15px;">
                <h4 style="color:white;margin-top:0;">State Overview: {sel_state}</h4>
                <div style="display:flex;flex-direction:column;gap:8px;">
                    <div style="display:flex;justify-content:space-between;">
                        <span>Crop Types:</span>
                        <span style="font-weight:bold;">{state_df['crop_type'].nunique()}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span>🌡 Avg Temp (NASA):</span>
                        <span style="font-weight:bold;">{state_df['temperature'].mean():.1f}°C</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span>💧 Avg Humidity (NASA):</span>
                        <span style="font-weight:bold;">{state_df['humidity'].mean():.1f}%</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        if not state_df.empty:
            dist = state_df.groupby("crop_type", as_index=False).agg(
                production=("production", "sum"),
                yield_efficiency=("yield_efficiency", "mean"),
            )
            dist["percentage"] = dist["production"] / dist["production"].sum() * 100
            dist = dist.sort_values("percentage", ascending=False)

            pie = alt.Chart(dist).mark_arc().encode(
                theta=alt.Theta("percentage:Q", stack=True),
                color=alt.Color("crop_type:N", scale=alt.Scale(scheme="category20"),
                                legend=alt.Legend(title="Crop Types")),
                tooltip=[
                    alt.Tooltip("crop_type:N",       title="Crop"),
                    alt.Tooltip("percentage:Q",       title="Percentage",  format=".1f"),
                    alt.Tooltip("production:Q",       title="Production",  format=",.0f"),
                    alt.Tooltip("yield_efficiency:Q", title="Avg Yield",   format=".2f"),
                ],
            ).properties(title=f"Crop Distribution in {sel_state}", height=400)
            st.altair_chart(pie, use_container_width=True)

    # Deep dive
    st.markdown('<div class="sub-header">Deep Dive: Data Relationships & Patterns</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        explore_state = st.selectbox("Select State:", ["All States"] + sorted(filtered["state"].unique()))
    with col2:
        explore_crop  = st.selectbox("Select Crop:",  ["All Crops"]  + sorted(filtered["crop_type"].unique()))

    explore_filtered = filtered.copy()
    if explore_state != "All States":
        explore_filtered = explore_filtered[explore_filtered["state"] == explore_state]
    if explore_crop  != "All Crops":
        explore_filtered = explore_filtered[explore_filtered["crop_type"] == explore_crop]

    if explore_filtered.empty:
        st.warning("No data for selected filters.")
        st.stop()

    time_data = explore_filtered.groupby("year").agg(
        production=("production",      "sum"),
        temperature=("temperature",    "mean"),
        humidity=("humidity",          "mean"),
        yield_efficiency=("yield_efficiency", "mean"),
    ).reset_index()

    # Check if we have multiple years to show timeline trends
    if len(time_data) > 1:
        st.altair_chart(
            alt.Chart(time_data).mark_line(point=True, color="#2E7D32").encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("production:Q", title="Production (MT)", scale=alt.Scale(zero=False)),
                tooltip=["year", "production"],
            ).properties(title="Production Over Time", height=300),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(
                alt.Chart(time_data).mark_line(point=True, color="#FF6B6B").encode(
                    x=alt.X("year:O", title="Year"),
                    y=alt.Y("temperature:Q", title="Temperature — NASA POWER (°C)", scale=alt.Scale(zero=False)),
                    tooltip=["year", "temperature"],
                ).properties(title="Temperature Over Time (NASA POWER)", height=250),
                use_container_width=True,
            )
        with col2:
            st.altair_chart(
                alt.Chart(time_data).mark_line(point=True, color="#42A5F5").encode(
                    x=alt.X("year:O", title="Year"),
                    y=alt.Y("humidity:Q", title="Humidity — NASA POWER (%)", scale=alt.Scale(zero=False)),
                    tooltip=["year", "humidity"],
                ).properties(title="Humidity Over Time (NASA POWER)", height=250),
                use_container_width=True,
            )

        # Climate anomalies
        st.markdown("---")
        st.markdown('<div class="sub-header">Climate Anomalies & Heatmap</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Temperature Anomaly (NASA POWER vs. Average)**")
            baseline_temp = filtered["temperature"].mean()
            anomaly_df = filtered.groupby("year")["temperature"].mean().reset_index()
            anomaly_df["anomaly"] = anomaly_df["temperature"] - baseline_temp

            st.altair_chart(
                alt.Chart(anomaly_df).mark_bar().encode(
                    x=alt.X("year:O", title="Year"),
                    y=alt.Y("anomaly:Q", title="Deviation from Baseline (°C)"),
                    color=alt.condition(
                        alt.datum.anomaly > 0, alt.value("#e45756"), alt.value("#1c91d4")
                    ),
                    tooltip=["year", alt.Tooltip("anomaly", format=".2f")],
                ).properties(height=350),
                use_container_width=True,
            )
            st.info("🔴 Red = hotter than average | 🔵 Blue = cooler than average")
        with col_b:
            st.markdown("**Correlation Matrix**")
            corr_matrix = filtered[
                ["temperature", "humidity", "production", "yield_efficiency"]
            ].corr().reset_index().melt(id_vars="index")

            heatmap = alt.Chart(corr_matrix).mark_rect().encode(
                x=alt.X("index:N", title=None),
                y=alt.Y("variable:N", title=None),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                                legend=alt.Legend(title="Correlation")),
                tooltip=[
                    alt.Tooltip("index", title="Variable 1"),
                    alt.Tooltip("variable", title="Variable 2"),
                    alt.Tooltip("value", title="Correlation", format=".2f"),
                ],
            ).properties(height=350)

            text_layer = heatmap.mark_text(baseline="middle").encode(
                text=alt.Text("value:Q", format=".2f"),
                color=alt.condition(alt.datum.value > 0.5, alt.value("white"), alt.value("black")),
            )
            st.altair_chart(heatmap + text_layer, use_container_width=True)
            st.info("🟦 +1 = strong positive | 🟥 −1 = strong negative | ⬜ 0 = no relationship")

    else:
        # Fallback view for single-year slices
        st.info(
            "ℹ️ Historical timeline trends are unavailable because the filtered selection contains only 1 year of data.")

        st.markdown("---")
        st.markdown('<div class="sub-header">Cross-Sectional Climate Heatmap</div>', unsafe_allow_html=True)

        # We can still compute a correlation matrix across different states/crops for this year
        valid_corr_cols = [c for c in ["temperature", "humidity", "production", "yield_efficiency"] if
                           c in filtered.columns]
        if len(filtered) > 2 and len(valid_corr_cols) > 1:
            corr_matrix = filtered[valid_corr_cols].corr().reset_index().melt(id_vars="index")
            heatmap = alt.Chart(corr_matrix).mark_rect().encode(
                x=alt.X("index:N", title=None),
                y=alt.Y("variable:N", title=None),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
                tooltip=["index", "variable", alt.Tooltip("value", format=".2f")]
            ).properties(height=300)
            st.altair_chart(heatmap + heatmap.mark_text().encode(text=alt.Text("value:Q", format=".2f")),
                            use_container_width=True)

    # ── Scatter plots & Climate Impact on Yield ─────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sub-header">Climate Impact on Yield (NASA POWER Data)</div>', unsafe_allow_html=True)

    # Clean missing variables before plotting scatter distributions
    scatter = explore_filtered.dropna(subset=["temperature", "yield_efficiency"])
    scatter_h = explore_filtered.dropna(subset=["humidity", "yield_efficiency"])

    col1, col2 = st.columns(2)
    with col1:
        if len(scatter) >= 2:
            base = alt.Chart(scatter).mark_circle(size=60).encode(
                x=alt.X("temperature:Q", title="Temperature — NASA (°C)", scale=alt.Scale(zero=False)),
                y=alt.Y("yield_efficiency:Q", title="Yield Efficiency (MT/Ha)"),
                color=alt.Color("crop_type:N"),
                tooltip=["state", "crop_type", "year", "temperature", "yield_efficiency"],
            ).properties(title="Temperature vs Yield", height=400)

            # Regression line requires at least 3 distinct variance distributions
            if len(scatter) > 2 and scatter["temperature"].nunique() > 1:
                reg = base.transform_regression("temperature", "yield_efficiency").mark_line(color="red", strokeWidth=3)
                st.altair_chart(base + reg, use_container_width=True)
            else:
                st.altair_chart(base, use_container_width=True)

            corr = scatter["temperature"].corr(scatter["yield_efficiency"])
            if not pd.isna(corr):
                st.metric("Temp–Yield Correlation (NASA)", f"{corr:.3f}", delta_color="off")
        else:
            st.caption("Not enough data items to render Temperature vs Yield distribution.")

    with col2:
        if len(scatter_h) >= 2:
            base_h = alt.Chart(scatter_h).mark_circle(size=60).encode(
                x=alt.X("humidity:Q", title="Humidity — NASA (%)", scale=alt.Scale(zero=False)),
                y=alt.Y("yield_efficiency:Q", title="Yield Efficiency (MT/Ha)"),
                color=alt.Color("crop_type:N"),
                tooltip=["state", "year", "humidity", "yield_efficiency"],
            ).properties(title="Humidity vs Yield", height=400)

            if len(scatter_h) > 2 and scatter_h["humidity"].nunique() > 1:
                reg_h = base_h.transform_regression("humidity", "yield_efficiency").mark_line(color="blue",
                                                                                              strokeWidth=3)
                st.altair_chart(base_h + reg_h, use_container_width=True)
            else:
                st.altair_chart(base_h, use_container_width=True)

            corr_h = scatter_h["humidity"].corr(scatter_h["yield_efficiency"])
            if not pd.isna(corr_h):
                st.metric("Humidity–Yield Correlation (NASA)", f"{corr_h:.3f}", delta_color="off")
        else:
            st.caption("Not enough data items to render Humidity vs Yield distribution.")

    # ── Heat Sensitivity Analysis ──────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sub-header">Heat Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.caption("Yield change (MT/Ha) per +1°C rise in temperature. Red = at-risk crop.")

    safe = explore_filtered.dropna(subset=["temperature", "yield_efficiency"])
    safe = safe.replace([np.inf, -np.inf], np.nan).dropna(subset=["yield_efficiency"])
    sens_rows = []

    # Requires data spread across different climate points to build regression slopes
    for crop in safe["crop_type"].unique():
        sub = safe[safe["crop_type"] == crop]
        if len(sub) > 2 and sub["temperature"].nunique() > 1:
            try:
                reg = LinearRegression().fit(sub[["temperature"]], sub["yield_efficiency"])
                sens_rows.append({"Crop": crop, "Sensitivity": reg.coef_[0]})
            except Exception:
                pass

    if sens_rows:
        sens_df = pd.DataFrame(sens_rows)
        st.altair_chart(
            alt.Chart(sens_df).mark_bar().encode(
                x=alt.X("Crop:N", sort="y"),
                y=alt.Y("Sensitivity:Q", title="Yield Change per +1°C (MT/Ha)"),
                color=alt.condition(
                    alt.datum.Sensitivity < 0, alt.value("#d32f2f"), alt.value("#388e3c")
                ),
                tooltip=["Crop", alt.Tooltip("Sensitivity", format=".4f")],
            ).properties(height=350),
            use_container_width=True,
        )
        st.info("🔴 Down (red) = yield drops when hotter | 🟢 Up (green) = heat-tolerant crop")
    else:
        st.warning(
            "⚠️ Linear calculations require variations in temperature data across records. Try expanding your dashboard filter ranges to multiple states or years.")

# =============================================================================
# TAB 3 — CLIMATE SIMULATION
# =============================================================================
else:
    st.markdown('<div class="sub-header">Climate Impact Simulation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1a472a;padding:15px;border-radius:10px;border-left:5px solid #2e8b57;
                color:white;margin-bottom:20px;">
        <b style="color:white;">Simulation Objective:</b> Explore how temperature and humidity changes
        might affect crop yields using real NASA POWER climate baselines.
    </div>""", unsafe_allow_html=True)

    sel_c1, sel_c2 = st.columns(2)
    with sel_c1:
        sim_state = st.selectbox("Select State:", sorted(filtered["state"].unique()), key="sim_state_select")
    with sel_c2:
        sim_crop  = st.selectbox("Select Crop:",  sorted(filtered["crop_type"].unique()), key="sim_crop_select")

    sim_data = filtered[
        (filtered["state"]     == sim_state) &
        (filtered["crop_type"] == sim_crop)
    ].sort_values("year")

    if len(sim_data) < 2:
        st.warning(f"Not enough data for {sim_crop} in {sim_state}. Please select another combination.")
        st.stop()

    # Show NASA baseline for this state
    nasa_baseline = sim_data[["year", "temperature", "humidity"]].drop_duplicates().sort_values("year")
    st.markdown(f"**NASA POWER Climate Baseline — {sim_state}**")
    st.dataframe(
        nasa_baseline.rename(columns={"temperature": "Temp (°C)", "humidity": "Humidity (%)"})
                     .set_index("year"),
        use_container_width=False,
    )

    st.markdown("---")
    st.markdown("### Simulation Parameters")

    param_c1, param_c2 = st.columns(2)
    with param_c1:
        temp_increase = st.slider("Temperature Increase (°C):",
                                  min_value=0.0, max_value=3.0, value=1.5, step=0.5)
    with param_c2:
        hum_change    = st.slider("Humidity Change (%):",
                                  min_value=-10.0, max_value=5.0, value=0.0, step=1.0)

    # Scenario comparison
    model_data = sim_data[["year", "temperature", "humidity", "production"]].dropna()

    if len(model_data) > 2:
        X = model_data[["temperature", "humidity"]]
        y = model_data["production"]
        model_simple = LinearRegression().fit(X, y)

        curr_vals   = [[model_data["temperature"].mean(), model_data["humidity"].mean()]]
        future_vals = [[curr_vals[0][0] + temp_increase,  curr_vals[0][1] + hum_change]]

        base_pred        = model_simple.predict(curr_vals)[0]
        sim_pred         = model_simple.predict(future_vals)[0]
        simple_change_pct = ((sim_pred - base_pred) / base_pred) * 100

        comparison_df = pd.DataFrame({
            "Scenario":   ["Baseline (NASA Current)", "Simulated (Future)"],
            "Production": [base_pred, sim_pred],
        })

        col_chart1, col_kpi = st.columns([2, 1])
        with col_chart1:
            st.altair_chart(
                alt.Chart(comparison_df).mark_bar().encode(
                    x=alt.X("Scenario:N", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Production:Q"),
                    color=alt.Color("Scenario:N", scale=alt.Scale(range=["#2E7D32", "#FF9800"])),
                ).properties(height=250, title="Projected Production Impact"),
                use_container_width=True,
            )
        with col_kpi:
            st.markdown("#### Impact Summary")
            st.metric("Baseline (NASA)",  f"{base_pred:,.0f} MT")
            st.metric("Simulated",        f"{sim_pred:,.0f} MT")
            st.metric("Net Change",       f"{simple_change_pct:+.2f}%",
                      delta_color="inverse" if simple_change_pct < 0 else "normal")

    # Time-series forecast
    st.markdown("---")
    st.markdown('<div class="sub-header">Time-Series Simulation (Linear Regression)</div>', unsafe_allow_html=True)
    st.caption("Projects the historical trend forward using Linear Regression with NASA POWER climate baselines.")

    reg_df = sim_data[["year", "production", "temperature", "humidity"]].dropna()

    if len(reg_df) > 3:
        X    = reg_df[["year", "temperature", "humidity"]]
        y    = reg_df["production"]
        model = LinearRegression().fit(X, y)
        r2   = model.score(X, y)

        last_year    = int(reg_df["year"].max())
        future_years = [last_year + i for i in range(1, 6)]

        base_temp = reg_df.tail(3)["temperature"].mean()
        base_hum  = reg_df.tail(3)["humidity"].mean()
        sim_temp  = base_temp + temp_increase
        sim_hum   = base_hum  + hum_change

        future_X = pd.DataFrame({
            "year":        future_years,
            "temperature": [sim_temp] * 5,
            "humidity":    [sim_hum]  * 5,
        })
        future_X["production"] = model.predict(future_X)
        future_X["Type"]       = "Forecast"

        hist_data             = reg_df[["year", "production"]].copy()
        hist_data["Type"]     = "Historical"
        full_data             = pd.concat([hist_data, future_X[["year", "production", "Type"]]])

        line = alt.Chart(full_data).mark_line(point=True).encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("production:Q", title="Production (MT)"),
            color=alt.Color("Type:N",
                            scale=alt.Scale(domain=["Historical", "Forecast"],
                                            range=["#2E7D32", "#FF9800"])),
            strokeDash=alt.condition(
                alt.datum.Type == "Forecast", alt.value([5, 5]), alt.value([0])
            ),
            tooltip=["year", "production", "Type"],
        ).properties(title=f"Projected Production — {sim_crop} in {sim_state}", height=350)

        st.altair_chart(line, use_container_width=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Temp Impact (per °C)",   f"{model.coef_[1]:,.0f} MT",
                      delta_color="inverse" if model.coef_[1] < 0 else "normal")
        with col_m2:
            st.metric("Humidity Impact (per %)", f"{model.coef_[2]:,.0f} MT")
        with col_m3:
            st.metric("Model R²", f"{r2:.3f}")

        trend_dir = "increase" if future_X["production"].iloc[-1] > hist_data["production"].iloc[-1] else "decrease"
        st.info(f"""
        **Simulation Summary (NASA POWER Baseline):**
        At {sim_temp:.1f}°C and {sim_hum:.1f}% humidity, production is predicted to
        **{trend_dir}** over the next 5 years.
        NASA baseline used: {base_temp:.1f}°C / {base_hum:.1f}% (3-year average).
        """)
    else:
        st.warning("Not enough data to train the simulation model (need at least 3 years).")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666;padding:2rem;">
    <p style="font-size:1.2rem;font-weight:bold;">🌾 Climate Impact on Food Availability Dashboard</p>
    <p>Developed for Final Year Project • Universiti Putra Malaysia</p>
    <p style="font-size:0.85rem;margin-top:0.5rem;">
        🛰️ Climate data sourced from <b>NASA POWER API</b>
        (T2M & RH2M parameters, Agroclimatology community)
    </p>
</div>
""", unsafe_allow_html=True)
