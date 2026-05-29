import streamlit as st
import numpy as np
import joblib

# Load model and scaler (CORRECT)
model = joblib.load("rain.pkl") # raindom forest

st.set_page_config(page_title="Rain Prediction", layout="centered")
st.title("🌍 Rain Prediction App")

# -------- User Inputs --------
Temperature = st.number_input("Temperature", min_value=0.0)
Humidity = st.number_input("Humidity", min_value=0.0)
Pressure = st.number_input("Pressure", min_value=0)
Windspeed = st.number_input("Windspeed", min_value=1)
Cloudcover = st.number_input("Cloudcover", min_value=0.0)
RainToday = st.number_input("RainToday", min_value=0,max_value=1)

# -------- Prediction --------
if st.button("Predict Tomorrow Rain"):
    input_data = np.array([[Temperature, Humidity, Pressure, Windspeed, Cloudcover, RainToday]])

    # ✅ Model uses predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.success(f"Predicted Rain: {prediction}")
    st.info(
        f"Confidence → Rain 0: {probability[0]:.2f}, "
        f"Rain 1: {probability[1]:.2f}"
    )


#st.write("Model:", type(model))
#st.write("Scaler:", type(scaler))
