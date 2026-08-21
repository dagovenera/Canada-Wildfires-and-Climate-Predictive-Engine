import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# STEP 1: CONFIGURATION & MODEL INGESTION
# ==========================================
st.set_page_config(
    page_title="Canada Wildfire Predict Engine",
    page_icon="🔥",
    layout="wide",
)

# Leverage Streamlit's caching mechanism so the model stays in memory
@st.cache_resource
def load_prediction_artifacts():
    # Load trained model and feature array
    return joblib.load("pipeline/canadian_wildfire_model.pkl")


try:
    artifacts = load_prediction_artifacts()
    model = artifacts["model"]
    feature_cols = artifacts["features"]
except FileNotFoundError:
    st.error(
        "❌ 'canadian_wildfire_model.pkl' not found. Please run training script first!"
    )
    st.stop()


# ==========================================
# STEP 2: USER INTERFACE LAYOUT & HEADERS
# ==========================================
st.title("🔥 Canadian Wildfire & Climate Predictive Engine")
st.markdown(
    """
    **Data Science MVP Portfolio Project** | *Domain Context: Environment and Climate Change in Canada*  
    This interface connects real-time user-defined meteorological metrics to an optimized **XGBoost Classifier** 
    to forecast localized, compounding wildfire risks across high-density Canadian forest structures.
    """
)
st.write("---")

# Split layout into two columns: Control Inputs (Left) and Machine Learning Output (Right)
col1, col2 = st.columns([1, 2], gap="large")


# =================================================
# STEP 3: CONTROLS & INTERACTIVE SLIDERS (Column 1)
# =================================================
with col1:
    st.subheader("📋 Current Meteorological Inputs")
    st.caption("Adjust sliders to simulate shifts in local weather.")

    # 1. Day-of Weather Inputs
    current_temp = st.slider("Current Temperature (°C)", -10.0, 45.0, 28.0, step=0.5)
    current_precip = st.slider(
        "Daily Precipitation (mm)", 0.0, 100.0, 0.0, step=0.1
    )
    wind_gust = st.slider("Max Wind Gust Speed (km/h)", 0.0, 120.0, 35.0, step=1.0)

    st.write("")
    st.subheader("⏳ Compound Historical Context")
    st.caption("Simulate multi-day drought metrics")

    # 2. Engineered Cumulative Inputs
    rolling_temp = st.slider(
        "7-Day Rolling Average Temp (°C)", -10.0, 45.0, 31.0, step=0.5
    )
    rolling_precip = st.slider(
        "7-Day Cumulative Rainfall (mm)", 0.0, 300.0, 4.0, step=1.0
    )


# ==========================================
# STEP 4: MODEL INFERENCE PIPELINE (Column 2)
# ==========================================
with col2:
    st.subheader("📊 Predictive Model Metrics")

    # Structure the real-time slider metrics to exactly match the model's feature matrix
    input_data = pd.DataFrame(
        [
            [
                current_temp,
                current_precip,
                wind_gust,
                rolling_temp,
                rolling_precip,
            ]
        ],
        columns=feature_cols,
    )

    # Execute Model Inference (Calculate Probability Vector)
    prediction_proba = model.predict_proba(input_data)[0][1]
    risk_percentage = prediction_proba * 100

    # Dynamic Alert Styling based on predictive scoring thresholds
    if risk_percentage < 35:
        status_color = "green"
        status_label = "LOW RISK PROFILE"
        st.success(f"✅ Environmental systems stable. Risk profile: **{status_label}**")
    elif risk_percentage < 70:
        status_color = "orange"
        status_label = "MODERATE/ELEVATED RISK"
        st.warning(f"⚠️ Atmospheric metrics priming. Risk profile: **{status_label}**")
    else:
        status_color = "red"
        status_label = "CRITICAL WILDFIRE DANGER"
        st.error(
            f"🚨 High probability of active ignition. Risk profile: **{status_label}**"
        )

    # Display the final prediction as a high-visibility metric
    st.metric(
        label="Calculated Wildfire Probability", value=f"{risk_percentage:.1f}%"
    )

    # Provide clear technical visibility into the feature parameters passed to XGBoost
    with st.expander("🛠️ View Raw JSON Data Payload passed to Model Vector"):
        st.json(input_data.to_dict(orient="records")[0])


# ==========================================
# STEP 5: FOOTER PORTFOLIO ANCHOR
# ==========================================
st.write("---")
st.caption(
    """
    Designed and engineered by **Dagoberto Venera-Ponton, PhD**.  
    Open-source code available on [GitHub](https://github.com/dagovenera/Canada-Wildfires-and-Climate-Predictive-Engine.git).
    """
)
