import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and encoder once at startup (cached so they don't reload on every interaction)
@st.cache_resource
def load_artifacts():
    with open("churn_rf_healthy_meals.pkl", "rb") as f:
        model = pickle.load(f)
    with open("churn_encoder_healthy_meals.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

age                = st.number_input("Age", min_value=18, max_value=100, value=35)
income_level       = st.radio("Income Level",  ["Low", "Medium", "High", "Very High"])
education          = st.radio("Education",     ["Graduate", "High School", "Other", "Post-Graduate"])
device_type        = st.radio("Device Type",   ["Desktop-only", "Mobile-only", "Multi-device"])
tech_comfort_score = st.number_input("Tech Comfort Score", min_value=1, max_value=10, value=5)

# Added: activity features used by your trained model
active_quarters     = st.number_input("Active Quarters (2022)", min_value=0, max_value=4, value=2)
total_sessions      = st.number_input("Total Sessions (2022)", min_value=0, value=20)
total_session_length = st.number_input("Total Session Length (2022 minutes)", min_value=0, value=300)
total_active_days   = st.number_input("Total Active Days (2022)", min_value=0, max_value=366, value=25)

if st.button("Predict"):

    # Build categorical DataFrame — column names must match encoder exactly
    raw = pd.DataFrame([{
        "INCOME_LEVEL": income_level,
        "EDUCATION":    education,
        "DEVICE_TYPE":  device_type,
    }])

    # Apply the saved encoder (transform only — never fit_transform)
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Added: all numeric model features
    avg_sessions_per_active_quarter = (
        total_sessions / active_quarters if active_quarters > 0 else 0
    )

    numeric_df = pd.DataFrame([{
        "AGE":                             age,
        "TECH_COMFORT_SCORE":              tech_comfort_score,
        "ACTIVE_QUARTERS":                 active_quarters,
        "TOTAL_SESSIONS":                  total_sessions,
        "TOTAL_SESSION_LENGTH":            total_session_length,
        "TOTAL_ACTIVE_DAYS":               total_active_days,
        "AVG_SESSIONS_PER_ACTIVE_QUARTER": avg_sessions_per_active_quarter,
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    # Keep training column order
    if hasattr(model, "feature_names_in_"):
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Column 1 = P(renewed), churn = 1 - renewed
    probability = model.predict_proba(input_df)[0][1]
    churn_probability = 1 - probability

    risk = "Low" if churn_probability < 0.4 else "Medium" if churn_probability < 0.6 else "High"

    st.metric("Renewal Probability", f"{probability:.2f}")
    st.metric("Churn Probability", f"{churn_probability:.2f}")
    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")
