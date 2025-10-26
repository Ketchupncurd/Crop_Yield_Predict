# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------
# Load model & artifacts
# -----------------------
MODEL_PATH = "models"
MODEL_FILE = os.path.join(MODEL_PATH, "lgbm_model.joblib")
ENCODERS_FILE = os.path.join(MODEL_PATH, "label_encoders.joblib")
FEATURES_FILE = os.path.join(MODEL_PATH, "features_list.joblib")

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_FILE)
        encoders = joblib.load(ENCODERS_FILE)
        features = joblib.load(FEATURES_FILE)
        return model, encoders, features
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

model, encoders, features = load_artifacts()

# -----------------------
# App Title
# -----------------------
st.title("🌾 Crop Yield Prediction App")
st.write("Predict crop yield based on agricultural and environmental factors.")

# -----------------------
# User Inputs
# -----------------------
if model is not None and encoders is not None and features is not None:
    st.header("Input Crop Details")

    # Example input fields
    year = st.number_input("Year", min_value=2000, max_value=2030, value=2024)
    area = st.number_input("Area (in hectares)", min_value=0.1, step=0.1)
    crop = st.selectbox("Crop Type", encoders["Crop"].classes_ if "Crop" in encoders else [])
    nitrogen = st.number_input("Nitrogen (Kg/Ha)", min_value=0.0)
    phosphorus = st.number_input("Phosphorus (Kg/Ha)", min_value=0.0)
    potassium = st.number_input("Potassium (Kg/Ha)", min_value=0.0)
    pest_score = st.slider("Pest Incidence Score", 0, 10, 5)

    if st.button("Predict Yield"):
        try:
            # Prepare input data
            input_data = pd.DataFrame([{
                "Year": year,
                "Area_Hectares": area,
                "Crop": crop,
                "Nitrogen_KgPerHa": nitrogen,
                "Phosphorus_KgPerHa": phosphorus,
                "Potassium_KgPerHa": potassium,
                "Pest_Incidence_Score": pest_score
            }])

            # Encode categorical variables
            for col, encoder in encoders.items():
                if col in input_data.columns:
                    input_data[col] = encoder.transform(input_data[col])

            # Ensure correct feature order
            input_data = input_data[features]

            # Predict
            prediction = model.predict(input_data)[0]
            st.success(f"✅ Predicted Crop Yield: {prediction:.2f} tonnes per hectare")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("Model files could not be loaded. Please check the models folder.")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("Developed using LightGBM, Optuna, and Streamlit.")
