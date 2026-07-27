import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_artifacts():
    with open("churn_rf_healthy_meals.pkl", "rb") as f:
        model = pickle.load(f)
    with open("churn_encoder_healthy_meals.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict renewal/churn likelihood.")

# --- defaults for reset ---
DEFAULTS = {
    "age": 35,
    "income_level": "Medium",
    "education": "Graduate",
    "device_type": "Multi-device",
    "tech_comfort_score": 5,
    "active_quarters": 2,
    "total_sessions": 20,
    "total_session_length": 300,
    "total_active_days": 25,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- reset button ---
if st.button("Reset to sample customer"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

# --- grouped sections ---
st.subheader("Demographics")
age = st.number_input("Age", min_value=18, max_value=100, key="age")
income_level = st.radio("Income Level", ["Low", "Medium", "High", "Very High"], key="income_level")
education = st.radio("Education", ["Graduate", "High School", "Other", "Post-Graduate"], key="education")
device_type = st.radio("Device Type", ["Desktop-only", "Mobile-only", "Multi-device"], key="device_type")
tech_comfort_score = st.number_input("Tech Comfort Score", min_value=1, max_value=10, key="tech_comfort_score")

st.subheader("2022 Activity")
active_quarters = st.number_input("Active Quarters (2022)", min_value=0, max_value=4, key="active_quarters")
total_sessions = st.number_input("Total Sessions (2022)", min_value=0, key="total_sessions")
total_session_length = st.number_input("Total Session Length (2022 minutes)", min_value=0, key="total_session_length")
total_active_days = st.number_input("Total Active Days (2022)", min_value=0, max_value=366, key="total_active_days")

if st.button("Predict"):
    income_map = {"Low": "low", "Medium": "medium", "High": "high", "Very High": "very high"}
    education_map = {
        "Graduate": "graduate",
        "High School": "high school",
        "Other": "other",
        "Post-Graduate": "post graduate",
    }

    raw = pd.DataFrame([{
        "EDUCATION": education_map[education],
        "INCOME_LEVEL": income_map[income_level],
        "DEVICE_TYPE": device_type,
    }])

    raw = raw[encoder.feature_names_in_]
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(encoder.feature_names_in_))

    avg_sessions_per_active_quarter = total_sessions / active_quarters if active_quarters > 0 else 0

    numeric_df = pd.DataFrame([{
        "AGE": age,
        "TECH_COMFORT_SCORE": tech_comfort_score,
        "ACTIVE_QUARTERS": active_quarters,
        "TOTAL_SESSIONS": total_sessions,
        "TOTAL_SESSION_LENGTH": total_session_length,
        "TOTAL_ACTIVE_DAYS": total_active_days,
        "AVG_SESSIONS_PER_ACTIVE_QUARTER": avg_sessions_per_active_quarter,
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    if hasattr(model, "feature_names_in_"):
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    renewed_prob = model.predict_proba(input_df)[0][1]
    churn_prob = 1 - renewed_prob

    risk = "Low" if churn_prob < 0.4 else "Medium" if churn_prob < 0.6 else "High"

    st.metric("Renewal Probability", f"{renewed_prob:.2f}")
    st.metric("Churn Probability", f"{churn_prob:.2f}")

    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")
