import streamlit as st
import sys
import os
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'source')))
from predict import predict_delay

st.set_page_config(page_title="Construction Delay Predictor", layout="centered")

st.title("🏗️ Construction Delay Predictor")
st.write("Enter your project details to see if it's at risk of delay.")


with st.form("delay_form"):
    project_type = st.selectbox("Project Type", ["Residential", "Commercial", "Infrastructure"])
    county = st.selectbox("County", ["Dublin", "Cork", "Galway", "Limerick", "Waterford", "Kilkenny"])
    planned_duration = st.number_input("Planned Duration (in days)", min_value=30, max_value=1000, value=180)
    progress = st.slider("Actual Progress (%)", min_value=0, max_value=100, value=60)
    rfis = st.number_input("Number of RFIs Raised", min_value=0, max_value=100, value=10)
    rain_days = st.slider("Rain Days So Far", min_value=0, max_value=30, value=5)

    submit = st.form_submit_button("Predict Delay")

if submit:
    input_data = {
        "Project_Type": project_type,
        "County": county,
        "Planned_Duration": planned_duration,
        "Actual_Progress (%)": progress,
        "RFIs": rfis,
        "Rain_Days": rain_days
    }

    result = predict_delay(input_data)

    st.subheader("📊 Prediction Result")
    st.markdown(f"**Status:** `{result['result']}`")
    st.markdown(f"**Confidence:** `{result['confidence']}`")
