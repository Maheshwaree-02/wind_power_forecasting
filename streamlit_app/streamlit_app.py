import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="WindVision AI",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "T1.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #07111f 0%,
        #0b1727 50%,
        #08101c 100%
    );
    color: white;
}

/* Main container */

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* Hide Streamlit UI */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Hero */

.hero {
    padding: 2rem;
    border-radius: 24px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,198,255,0.12);
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}

/* KPI cards */

.metric-card {
    background: rgba(255,255,255,0.04);
    border-radius: 22px;
    padding: 1.2rem;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    transition: 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border: 1px solid rgba(0,198,255,0.4);
}

.metric-title {
    color: #8ba3c7;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: white;
}

/* Section cards */

.section-card {
    background: rgba(255,255,255,0.03);
    border-radius: 24px;
    padding: 1.2rem;
    border: 1px solid rgba(255,255,255,0.06);
    margin-top: 1rem;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    if DATA_PATH.exists():

        df = pd.read_csv(DATA_PATH)

    else:

        dates = pd.date_range(
            start="2024-01-01",
            periods=5000,
            freq="H"
        )

        np.random.seed(42)

        wind = np.random.uniform(2, 20, len(dates))

        theoretical = wind ** 3 * 4

        active = theoretical * np.random.uniform(
            0.7,
            1.0,
            len(dates)
        )

        direction = np.random.uniform(0, 360, len(dates))

        df = pd.DataFrame({
            "Date/Time": dates,
            "LV ActivePower (kW)": active,
            "Wind Speed (m/s)": wind,
            "Theoretical_Power_Curve (KWh)": theoretical,
            "Wind Direction (°)": direction
        })

    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    df.rename(columns={
        'LV ActivePower (kW)': 'ActivePower',
        'Wind Speed (m/s)': 'WindSpeed',
        'Theoretical_Power_Curve (KWh)': 'TheoreticalPower',
        'Wind Direction (°)': 'WindDirection'
    }, inplace=True)

    df.set_index("Date/Time", inplace=True)

    # Features

    df["Hour"] = df.index.hour
    df["Month"] = df.index.month

    df["Lag_1"] = df["ActivePower"].shift(1)

    df["Rolling_24"] = (
        df["ActivePower"]
        .rolling(24)
        .mean()
    )

    df.dropna(inplace=True)

    return df


df = load_data()

# =====================================================
# TRAIN MODEL
# =====================================================

@st.cache_resource
def train_model():

    features = [
        "WindSpeed",
        "TheoreticalPower",
        "WindDirection",
        "Hour",
        "Month",
        "Lag_1",
        "Rolling_24"
    ]

    X = df[features]
    y = df["ActivePower"]

    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)

    return model, pred, y_test, r2, features


model, pred, y_test, r2, FEATURES = train_model()

# =====================================================
# NAVBAR
# =====================================================

selected = option_menu(
    menu_title=None,
    options=[
        "Overview",
        "Forecasting",
        "ML Engine",
        "Simulator"
    ],
    icons=[
        "activity",
        "graph-up",
        "cpu",
        "wind"
    ],
    orientation="horizontal",
    default_index=0
)

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="hero">

<h1 style="margin-bottom:0;">
WindVision AI
</h1>

<p style="color:#8ba3c7;font-size:1.05rem;">
Enterprise Renewable Forecast Intelligence Platform
</p>

</div>
""", unsafe_allow_html=True)

# =====================================================
# KPI ROW
# =====================================================

col1, col2, col3, col4 = st.columns(4)

avg_power = df["ActivePower"].mean() / 1000

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Output</div>
        <div class="metric-value">{avg_power:.2f} MW</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Forecast Accuracy</div>
        <div class="metric-value">{r2:.2%}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Wind Speed</div>
        <div class="metric-value">{df['WindSpeed'].mean():.1f} m/s</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total = df["ActivePower"].sum() / 1000

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Energy Generated</div>
        <div class="metric-value">{total:,.0f} MWh</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# OVERVIEW
# =====================================================

if selected == "Overview":

    st.markdown("## Live Energy Intelligence")

    daily = (
        df["ActivePower"]
        .resample("D")
        .mean()
        .reset_index()
    )

    fig = px.area(
        daily,
        x="Date/Time",
        y="ActivePower"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500
    )

    st.plotly_chart(fig, width='stretch')

# =====================================================
# FORECASTING
# =====================================================

elif selected == "Forecasting":

    st.markdown("## Forecast Analytics")

    forecast_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": pred
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=forecast_df["Actual"][:300],
        mode="lines",
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        y=forecast_df["Predicted"][:300],
        mode="lines",
        name="Predicted"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500
    )

    st.plotly_chart(fig, width='stretch')

# =====================================================
# ML ENGINE
# =====================================================

elif selected == "ML Engine":

    st.markdown("## ML Intelligence")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500
    )

    st.plotly_chart(fig, width='stretch')

# =====================================================
# SIMULATOR
# =====================================================

elif selected == "Simulator":

    st.markdown("## Live Forecast Simulator")

    col1, col2 = st.columns(2)

    with col1:
        wind_speed = st.slider(
            "Wind Speed",
            0.0,
            25.0,
            10.0
        )

        wind_dir = st.slider(
            "Wind Direction",
            0,
            360,
            180
        )

    with col2:
        hour = st.slider(
            "Hour",
            0,
            23,
            12
        )

        month = st.slider(
            "Month",
            1,
            12,
            6
        )

    theoretical = wind_speed ** 3 * 4

    lag = df["ActivePower"].iloc[-1]

    rolling = df["Rolling_24"].iloc[-1]

    input_df = pd.DataFrame([[
        wind_speed,
        theoretical,
        wind_dir,
        hour,
        month,
        lag,
        rolling
    ]], columns=FEATURES)

    prediction = model.predict(input_df)[0]

    st.metric(
        "Predicted Power Output",
        f"{prediction:.2f} kW"
    )

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        gauge={
            'axis': {'range': [0, 4000]}
        }
    ))

    gauge.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350
    )

    st.plotly_chart(gauge, width='stretch')
