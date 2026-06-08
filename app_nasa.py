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
# State coordinates
# -------------------------
STATE_COORDS = {
    "Johor": (1.4927, 103.7414),
    "Kedah": (6.1254, 100.3678),
    "Kelantan": (6.1252, 102.2382),
    "Melaka": (2.1896, 102.2501),
    "Negeri Sembilan": (2.7254, 101.9420),
    "Pahang": (3.8079, 102.5485),
    "Pulau Pinang": (5.4164, 100.3327),
    "Perak": (4.5975, 101.0901),
    "Perlis": (6.4425, 100.2083),
    "Sabah": (5.9804, 116.0735),
    "Sarawak": (1.5533, 110.3593),
    "Selangor": (3.0738, 101.5183),
    "Terengganu": (5.3333, 103.1333),
    "W.P. Kuala Lumpur": (3.1390, 101.6869),
    "W.P. Labuan": (5.2830, 115.2340),
    "Malaysia": (4.2105, 101.9758),
}


# -------------------------
# Helper functions
# -------------------------
def clean_numeric(series: pd.Series) -> pd.Series:
    """Robust numeric cleaning using fast regex replacement."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace("", np.nan),
        errors="coerce",
    )


def normalise_state(name: str) -> str:
    """Standardize state names safely."""
    return str(name).strip().title() if pd.notna(name) else "Unknown"


# -------------------------
# NASA climate loader
# -------------------------
@st.cache_data
def load_nasa_climate():
    """Load pre-fetched NASA POWER climate CSV."""
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
            climate["state"] = climate["state"].apply(normalise_state)
            climate["year"] = climate["year"].astype(int)
            return climate

    uploaded = st.sidebar.file_uploader(
        "Upload climate_state_2017_2022.csv",
        type="csv",
        key="nasa_climate_uploader",
    )
    if uploaded is not None:
        climate = pd.read_csv(uploaded)
        climate["state"] = climate["state"].apply(normalise_state)
        climate["year"] = climate["year"].astype(int)
        st.sidebar.success("✅ NASA climate data loaded!")
        return climate

    return None


# -------------------------
# Main data loader
# -------------------------
@st.cache_data
def load_data():
    """Core data processing pipeline."""
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
        uploaded_file = st.sidebar.file_uploader("Upload crops_state.csv", type="csv", key="crops_uploader")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding="latin1")
            st.sidebar.success("Crops file uploaded!")
        else:
            st.sidebar.info("Using sample data for demonstration.")
            return create_sample_data()
    else:
        df = pd.read_csv(file_path, encoding="latin1")

    df.columns = [c.strip().lower() for c in df.columns]

    if "state" in df.columns:
        df["state"] = df["state"].apply(normalise_state)

    if "crop_type" in df.columns:
        df["crop_type"] = df["crop_type"].astype(str).str.strip().str.title()
        crop_mapping = {
            "Cash_Crops": "Corn", "Industrial_Crops": "Palm Oil",
            "Cash_crops": "Corn", "Industrial_crops": "Palm Oil",
            "Cash Crops": "Corn", "Industrial Crops": "Palm Oil",
        }
        df["crop_type"] = df["crop_type"].replace(crop_mapping)

    for col in ["production", "planted_area"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # Remove conflicting columns before merge
    df = df.drop(columns=[c for c in ["temperature", "humidity"] if c in df.columns])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
        df["year"] = df["date"].dt.year.fillna(2017).astype(int)
        df["month"] = df["date"].dt.month.fillna(1).astype(int)
    elif "year" not in df.columns:
        df["year"] = 2017

    nasa = load_nasa_climate()

    if nasa is not None:
        nasa = nasa.rename(columns={"mean_temp_c": "temperature", "mean_humidity_pct": "humidity"})
        df = df.merge(nasa[["state", "year", "temperature", "humidity"]], on=["state", "year"], how="left")

        missing_clim = df["temperature"].isna().sum()
        if missing_clim == 0:
            st.sidebar.success(f"✅ NASA climate merged seamlessly.")
        else:
            st.sidebar.warning(f"⚠️ NASA climate missing for {missing_clim} rows.")
    else:
        df["temperature"], df["humidity"] = np.nan, np.nan
        st.sidebar.error("❌ NASA climate dataset missing.")

    if {"production", "planted_area"}.issubset(df.columns):
        # Safe division avoiding divide-by-zero warnings
        df["yield_efficiency"] = np.divide(
            df["production"],
            df["planted_area"],
            out=np.zeros_like(df["production"]),
            where=df["planted_area"] > 0
        )
        df["yield_efficiency"] = df["yield_efficiency"].replace(0, np.nan)

    df["lat"] = df["state"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[0])
    df["lon"] = df["state"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[1])

    return df


def create_sample_data():
    """Generates fallback synthetic data."""
    states = ["Johor", "Kedah", "Kelantan", "Melaka", "Selangor"]
    crops = ["Corn", "Palm Oil", "Vegetables", "Fruits", "Paddy"]
    years = [2017, 2018, 2019, 2020, 2021, 2022]
    data = []
    for state in states:
        for crop in crops:
            for year in years:
                data.append({
                    "state": state, "crop_type": crop, "year": year,
                    "production": np.random.randint(10000, 100000) * (1 + 0.05 * (year - 2017)),
                    "planted_area": np.random.randint(1000, 10000),
                    "temperature": np.random.uniform(25, 32) + (year - 2017) * 0.1,
                    "humidity": np.random.uniform(70, 90),
                    "lat": 4.0 + np.random.uniform(-1, 1),
                    "lon": 102.0 + np.random.uniform(-2, 2),
                })
    df = pd.DataFrame(data)
    df["yield_efficiency"] = df["production"] / df["planted_area"]
    return df


# ── Load and Prep ─────────────────────────────────────────────────────────────
df = load_data()
if df.empty:
    st.error("No valid data available. Please check your data sources.")
    st.stop()

# -------------------------
# Sidebar configurations
# -------------------------
st.sidebar.header("Dashboard Controls")
tab_selection = st.sidebar.radio("Select Dashboard View:",
                                 ["📊 Summary & Overview", "🔍 Trend Analysis", "🎯 Climate Simulation"])

state_options = sorted([s for s in df["state"].dropna().unique() if s.lower() != "malaysia"])
crop_options = sorted(df["crop_type"].dropna().unique())

states_selected = st.sidebar.multiselect("Select State(s):", options=state_options, default=state_options)
crops_selected = st.sidebar.multiselect("Select Crop Type(s):", options=crop_options, default=crop_options)

min_year = int(df["year"].min()) if df["year"].notnull().any() else 2017
max_year = int(df["year"].max()) if df["year"].notnull().any() else 2022

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
    st.warning("No data found for the current filter selection.")
    st.stop()

filtered = filtered.sort_values(["state", "crop_type", "year"])
st.sidebar.info(f"Showing {len(filtered)} records")

# =============================================================================
# TAB 1 — SUMMARY & OVERVIEW
# =============================================================================
if tab_selection == "📊 Summary & Overview":
    st.markdown('<div class="sub-header">National Overview & Key Insights</div>', unsafe_allow_html=True)

    available_years = sorted(filtered["year"].unique())
    prod_delta = yield_delta = temp_delta = 0
    if len(available_years) >= 2:
        latest, prev = available_years[-1], available_years[-2]
        d_latest = filtered[filtered["year"] == latest]
        d_prev = filtered[filtered["year"] == prev]

        if d_prev["production"].sum() > 0:
            prod_delta = ((d_latest["production"].sum() - d_prev["production"].sum()) / d_prev[
                "production"].sum()) * 100
        if d_prev["yield_efficiency"].mean() > 0:
            yield_delta = ((d_latest["yield_efficiency"].mean() - d_prev["yield_efficiency"].mean()) / d_prev[
                "yield_efficiency"].mean()) * 100
        temp_delta = d_latest["temperature"].mean() - d_prev["temperature"].mean()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Production", f"{filtered['production'].sum():,.0f} MT", f"{prod_delta:+.1f}% YoY")
    kpi2.metric("Avg Yield Efficiency", f"{filtered['yield_efficiency'].mean():.2f} MT/Ha", f"{yield_delta:+.1f}% YoY")
    kpi3.metric("Avg Temp (NASA)", f"{filtered['temperature'].mean():.1f}°C", f"{temp_delta:+.1f}°C",
                delta_color="inverse")
    kpi4.metric("Total Planted Area", f"{filtered['planted_area'].sum():,.0f} Ha", "Cumulative")

    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
    kpi5.metric("Avg Humidity (NASA)", f"{filtered['humidity'].mean():.1f}%")
    kpi6.metric("States Covered", str(filtered["state"].nunique()))
    kpi7.metric("Crop Types", str(filtered["crop_type"].nunique()))
    kpi8.metric("Years of Data", str(filtered["year"].nunique()))

    st.info("🛰️ **Data Source:** NASA POWER API (T2M = Air Temp, RH2M = Rel. Humidity) — Aggregated annual means.")

    st.markdown('<div class="sub-header">Top Performing States</div>', unsafe_allow_html=True)
    state_perf = filtered.groupby("state").agg(
        production=("production", "sum"),
        yield_efficiency=("yield_efficiency", "mean"),
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean")
    ).reset_index()

    st.markdown("**Top 5 States by Production**")
    for col, (_, row) in zip(st.columns(5), state_perf.nlargest(5, "production").iterrows()):
        col.markdown(f"""
        <div style="background:#e8f5e9;padding:12px;border-radius:8px;text-align:center;border-left:4px solid #4CAF50;">
            <div style="font-weight:bold;color:#1a472a;">{row['state']}</div>
            <div style="font-size:1.1rem;font-weight:bold;">{row['production']:,.0f} MT</div>
            <div style="font-size:0.8rem;color:#555;">{row['temperature']:.1f}°C | {row['humidity']:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Top 5 States by Yield Efficiency**")
    for col, (_, row) in zip(st.columns(5), state_perf.nlargest(5, "yield_efficiency").iterrows()):
        col.markdown(f"""
        <div style="background:#e3f2fd;padding:12px;border-radius:8px;text-align:center;border-left:4px solid #2196F3;">
            <div style="font-weight:bold;color:#0d47a1;">{row['state']}</div>
            <div style="font-size:1.1rem;font-weight:bold;">{row['yield_efficiency']:.2f} MT/Ha</div>
            <div style="font-size:0.8rem;color:#555;">{row['temperature']:.1f}°C | {row['humidity']:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sub-header">Production Distribution Map</div>', unsafe_allow_html=True)

    # Safely construct map definitions avoiding NaN values
    map_df = state_perf.copy()
    map_df["lat"] = map_df["state"].map(lambda x: STATE_COORDS.get(x, (np.nan, np.nan))[0])
    map_df["lon"] = map_df["state"].map(lambda x: STATE_COORDS.get(x, (np.nan, np.nan))[1])
    map_df = map_df.dropna(subset=["lat", "lon"])

    if not map_df.empty:
        max_p, min_p = map_df["production"].max(), map_df["production"].min()
        map_df["radius"] = ((map_df["production"] - min_p) / (max_p - min_p + 1e-9)) * 50000 + 15000
        map_df["color"] = map_df["production"].apply(
            lambda p: [0, int(100 + ((p - min_p) / (max_p - min_p + 1e-9)) * 155), 0, 180])

        st.pydeck_chart(pdk.Deck(
            map_style="light",
            initial_view_state=pdk.ViewState(latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=5.5),
            layers=[
                pdk.Layer("ScatterplotLayer", map_df, get_position="[lon, lat]", get_radius="radius",
                          get_fill_color="color", pickable=True, auto_highlight=True),
                pdk.Layer("TextLayer", map_df, get_position="[lon, lat]", get_text="state", get_color=[0, 0, 0, 255],
                          get_size=14, font_weight="bold", get_alignment_baseline="'center'")
            ],
            tooltip={
                "text": "{state}\nProduction: {production} MT\nAvg Yield: {yield_efficiency} MT/Ha\nTemp: {temperature}°C"}
        ))

# =============================================================================
# TAB 2 — TREND ANALYSIS
# =============================================================================
elif tab_selection == "🔍 Trend Analysis":
    st.markdown('<div class="sub-header">Crop Distribution Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    sel_state = col1.selectbox("Select State:", filtered["state"].unique(), key="dist_state")
    state_df = filtered[filtered["state"] == sel_state]

    if not state_df.empty:
        col1.markdown(f"""
        <div style="background:#1a472a;padding:15px;border-radius:10px;color:white;">
            <h4 style="color:white;margin-top:0;">{sel_state} Overview</h4>
            <p><b>Crop Types:</b> {state_df['crop_type'].nunique()}<br>
            <b>Avg Temp:</b> {state_df['temperature'].mean():.1f}°C<br>
            <b>Avg Humidity:</b> {state_df['humidity'].mean():.1f}%</p>
        </div>""", unsafe_allow_html=True)

        dist = state_df.groupby("crop_type", as_index=False)["production"].sum()
        dist["percentage"] = (dist["production"] / dist["production"].sum()) * 100

        pie = alt.Chart(dist).mark_arc().encode(
            theta=alt.Theta("percentage:Q"), color=alt.Color("crop_type:N"),
            tooltip=["crop_type", alt.Tooltip("percentage:Q", format=".1f"), "production"]
        ).properties(title=f"Crop Distribution in {sel_state}", height=350)
        col2.altair_chart(pie, use_container_width=True)

    st.markdown('<div class="sub-header">Deep Dive: Data Relationships & Patterns</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    explore_state = c1.selectbox("Filter State:", ["All States"] + list(filtered["state"].unique()))
    explore_crop = c2.selectbox("Filter Crop:", ["All Crops"] + list(filtered["crop_type"].unique()))

    exp_df = filtered.copy()
    if explore_state != "All States": exp_df = exp_df[exp_df["state"] == explore_state]
    if explore_crop != "All Crops":  exp_df = exp_df[exp_df["crop_type"] == explore_crop]

    time_data = exp_df.groupby("year").mean(numeric_only=True).reset_index()

    if len(time_data) > 1:
        st.altair_chart(alt.Chart(time_data).mark_line(point=True, color="#2E7D32").encode(
            x="year:O", y=alt.Y("production:Q", scale=alt.Scale(zero=False)), tooltip=["year", "production"]
        ).properties(height=300, title="Production Over Time").interactive(), use_container_width=True)

        t_col1, t_col2 = st.columns(2)
        t_col1.altair_chart(alt.Chart(time_data).mark_line(point=True, color="#FF6B6B").encode(
            x="year:O", y=alt.Y("temperature:Q", scale=alt.Scale(zero=False)), tooltip=["year", "temperature"]
        ).properties(height=250, title="Temperature Over Time").interactive(), use_container_width=True)

        t_col2.altair_chart(alt.Chart(time_data).mark_line(point=True, color="#42A5F5").encode(
            x="year:O", y=alt.Y("humidity:Q", scale=alt.Scale(zero=False)), tooltip=["year", "humidity"]
        ).properties(height=250, title="Humidity Over Time").interactive(), use_container_width=True)

        st.markdown('<div class="sub-header">Climate Anomalies & Heatmap</div>', unsafe_allow_html=True)
        a_col, b_col = st.columns(2)

        baseline = exp_df["temperature"].mean()
        time_data["anomaly"] = time_data["temperature"] - baseline
        a_col.altair_chart(alt.Chart(time_data).mark_bar().encode(
            x="year:O", y="anomaly:Q",
            color=alt.condition(alt.datum.anomaly > 0, alt.value("#e45756"), alt.value("#1c91d4")),
            tooltip=["year", alt.Tooltip("anomaly", format=".2f")]
        ).properties(height=350, title="Temp Anomaly vs Average"), use_container_width=True)

        corr_mat = exp_df[["temperature", "humidity", "production", "yield_efficiency"]].corr().reset_index().melt(
            "index")
        heatmap = alt.Chart(corr_mat).mark_rect().encode(
            x="index:N", y="variable:N", color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=["index", "variable", alt.Tooltip("value", format=".2f")]
        ).properties(height=350, title="Correlation Matrix")
        text = heatmap.mark_text(baseline="middle").encode(text=alt.Text("value:Q", format=".2f"),
                                                           color=alt.value("white"))
        b_col.altair_chart(heatmap + text, use_container_width=True)
    else:
        st.info("ℹ️ Historical timeline trends unavailable (requires multiple years of filtered data).")

    st.markdown('<div class="sub-header">Climate Impact on Yield</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)

    clean_temp = exp_df.dropna(subset=["temperature", "yield_efficiency"])
    if len(clean_temp) > 2:
        scatter_t = alt.Chart(clean_temp).mark_circle(size=60).encode(
            x=alt.X("temperature:Q", scale=alt.Scale(zero=False)), y="yield_efficiency:Q", color="crop_type:N",
            tooltip=["state", "crop_type", "temperature", "yield_efficiency"]
        ).properties(height=350, title="Temp vs Yield").interactive()

        if clean_temp["temperature"].nunique() > 1:
            reg_t = scatter_t.transform_regression("temperature", "yield_efficiency").mark_line(color="red")
            s1.altair_chart(scatter_t + reg_t, use_container_width=True)
        else:
            s1.altair_chart(scatter_t, use_container_width=True)

    clean_hum = exp_df.dropna(subset=["humidity", "yield_efficiency"])
    if len(clean_hum) > 2:
        scatter_h = alt.Chart(clean_hum).mark_circle(size=60).encode(
            x=alt.X("humidity:Q", scale=alt.Scale(zero=False)), y="yield_efficiency:Q", color="crop_type:N",
            tooltip=["state", "crop_type", "humidity", "yield_efficiency"]
        ).properties(height=350, title="Humidity vs Yield").interactive()

        if clean_hum["humidity"].nunique() > 1:
            reg_h = scatter_h.transform_regression("humidity", "yield_efficiency").mark_line(color="blue")
            s2.altair_chart(scatter_h + reg_h, use_container_width=True)
        else:
            s2.altair_chart(scatter_h, use_container_width=True)

    st.markdown('<div class="sub-header">Heat Sensitivity Analysis</div>', unsafe_allow_html=True)
    sens_rows = []
    for crop in clean_temp["crop_type"].unique():
        sub = clean_temp[clean_temp["crop_type"] == crop]
        if len(sub) > 2 and sub["temperature"].nunique() > 1:
            try:
                m = LinearRegression().fit(sub[["temperature"]], sub["yield_efficiency"])
                sens_rows.append({"Crop": crop, "Sensitivity": m.coef_[0]})
            except:
                pass

    if sens_rows:
        st.altair_chart(alt.Chart(pd.DataFrame(sens_rows)).mark_bar().encode(
            x=alt.X("Crop:N", sort="y"), y="Sensitivity:Q",
            color=alt.condition(alt.datum.Sensitivity < 0, alt.value("#d32f2f"), alt.value("#388e3c")),
            tooltip=["Crop", alt.Tooltip("Sensitivity", format=".4f")]
        ).properties(height=300), use_container_width=True)
        st.caption("Yield change (MT/Ha) per +1°C rise. Red = Downward trend. Green = Heat tolerant.")
    else:
        st.warning("Insufficient temperature variance to calculate heat sensitivity metrics.")

# =============================================================================
# TAB 3 — CLIMATE SIMULATION
# =============================================================================
else:
    st.markdown('<div class="sub-header">Climate Impact Simulation</div>', unsafe_allow_html=True)

    sim_c1, sim_c2 = st.columns(2)
    sim_state = sim_c1.selectbox("Select State:", sorted(filtered["state"].unique()))
    sim_crop = sim_c2.selectbox("Select Crop:", sorted(filtered["crop_type"].unique()))

    sim_data = filtered[(filtered["state"] == sim_state) & (filtered["crop_type"] == sim_crop)].dropna(
        subset=["temperature", "humidity", "production"])

    if len(sim_data) < 3 or sim_data["temperature"].nunique() < 2:
        st.warning(
            f"Not enough historical variance to model {sim_crop} in {sim_state}. Need at least 3 distinct records.")
        st.stop()

    st.markdown("### Parameters")
    p1, p2 = st.columns(2)
    t_inc = p1.slider("Temp Increase (°C):", 0.0, 3.0, 1.5, 0.5)
    h_chg = p2.slider("Humidity Change (%):", -10.0, 5.0, 0.0, 1.0)

    X = sim_data[["temperature", "humidity"]]
    y = sim_data["production"]

    try:
        model = LinearRegression().fit(X, y)
        base_t, base_h = X["temperature"].mean(), X["humidity"].mean()
        base_pred = model.predict([[base_t, base_h]])[0]
        sim_pred = model.predict([[base_t + t_inc, base_h + h_chg]])[0]

        st.markdown("---")
        res1, res2, res3 = st.columns(3)
        res1.metric("Baseline Output", f"{base_pred:,.0f} MT")
        res2.metric("Simulated Output", f"{sim_pred:,.0f} MT")
        pct_change = ((sim_pred - base_pred) / base_pred) * 100
        res3.metric("Net Change", f"{pct_change:+.2f}%", delta_color="inverse" if pct_change < 0 else "normal")

        st.info(
            f"**Insight:** A {t_inc}°C rise and {h_chg}% humidity shift alters baseline production by roughly {pct_change:.1f}%. Model R² Score: {model.score(X, y):.2f}.")
    except Exception as e:
        st.error("Simulation failed due to mathematically unstable baseline data (collinearity).")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666;padding:2rem;">
    <p style="font-size:1.2rem;font-weight:bold;">🌾 Climate Impact Dashboard</p>
    <p style="font-size:0.85rem;margin-top:0.5rem;">
        🛰️ Climate data sourced from <b>NASA POWER API</b>
    </p>
</div>
""", unsafe_allow_html=True)
