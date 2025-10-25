# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------
# Load model & artifacts
# -----------------------
MODEL_PATH = "models"  # Folder where model and encoders are saved
# NOTE: Ensure these files exist in a 'models' directory:
# lgbm_model.joblib, label_encoders.joblib, features_list.joblib
MODEL_FILE = os.path.join(MODEL_PATH, "lgbm_model.joblib")
ENCODERS_FILE = os.path.join(MODEL_PATH, "label_encoders.joblib")
FEATURES_FILE = os.path.join(MODEL_PATH, "features_list.joblib")

@st.cache_data(show_spinner=False)
def load_artifacts():
    """Load the model, encoders, and feature list."""
    try:
        model = joblib.load(MODEL_FILE)
        encoders = joblib.load(ENCODERS_FILE)  # dict of LabelEncoder objects
        features = joblib.load(FEATURES_FILE)  # list of feature names (exact order)
        return model, encoders, features
    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure the 'models' directory contains '{os.path.basename(MODEL_FILE)}', '{os.path.basename(ENCODERS_FILE)}', and '{os.path.basename(FEATURES_FILE)}'.")
        # Return dummy values to allow the rest of the app to render for debugging
        return None, {'Crop': type('DummyEncoder', (object,), {'classes_': ['DummyCrop1', 'DummyCrop2']})()}, ['DummyFeature']

model, label_encoders, FEATURES = load_artifacts()

# Check if artifacts loaded successfully before continuing
if model is None:
    st.stop() 

# -----------------------
# Helper functions
# -----------------------
def encode_category(col_name, value):
    """Encode a categorical value using saved LabelEncoder.
        If unseen label, return -1 (LightGBM can handle negative)."""
    le = label_encoders.get(col_name)
    if le is None:
        # This case should be caught by load_artifacts if features_list is correct
        raise ValueError(f"No encoder found for {col_name}")
    if value in le.classes_:
        # LightGBM prefers integers for categorical features (which LabelEncoder provides)
        return int(le.transform([value])[0])
    else:
        # Unseen category - return -1 or a value outside the known range
        return -1

def make_input_df(user_inputs):
    """Build a 1-row DataFrame with the same feature names and order as training."""
    # Ensure the dictionary includes all required features in the correct order
    row = {feat: user_inputs.get(feat) for feat in FEATURES}
    return pd.DataFrame([row])

# -----------------------
# Streamlit UI Setup
# -----------------------
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌾 Maharashtra Oilseed Yield Predictor")
st.markdown(
    "Enter environmental, soil, and fertilizer data, then click **Predict Yield** to estimate the output (in kg/ha)."
)

st.sidebar.header("🔧 Input Controls")

# -----------------------
# Numeric Inputs
# -----------------------
year = st.sidebar.number_input("Year", min_value=2000, max_value=2030, value=2023, step=1)
area_hectares = st.sidebar.number_input("Area (Hectares)", min_value=0.0, max_value=10000.0, value=100.0, step=1.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 1200.0, 200.0, step=1.0)
min_temp = st.sidebar.slider("Min Temperature (°C)", -5.0, 40.0, 20.0, step=0.1)
max_temp = st.sidebar.slider("Max Temperature (°C)", -5.0, 50.0, 30.0, step=0.1)
temp_range = st.sidebar.slider("Temperature Range (°C)", 0.0, 40.0, 10.0, step=0.1)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0, step=1.0)
soil_ph = st.sidebar.slider("Soil pH", 4.0, 9.0, 7.0, step=0.01)
organic_carbon = st.sidebar.slider("Organic Carbon (%)", 0.0, 10.0, 1.0, step=0.01)
npk_total = st.sidebar.number_input("NPK Total (kg/ha)", min_value=0.0, max_value=5000.0, value=200.0, step=1.0)
fert_total = st.sidebar.number_input("Fertilizer Total (Urea+DAP+Potash, kg/ha)", min_value=0.0, max_value=1000.0, value=200.0, step=1.0)
nitrogen = st.sidebar.number_input("Nitrogen (Kg/Ha)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
phosphorus = st.sidebar.number_input("Phosphorus (Kg/Ha)", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
potassium = st.sidebar.number_input("Potassium (Kg/Ha)", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
pest_score = st.sidebar.slider("Pest Incidence Score (1–10)", 1, 10, 5)
urea = st.sidebar.number_input("Urea Applied (Kg/Ha)", min_value=0.0, max_value=1000.0, value=200.0, step=1.0)
dap = st.sidebar.number_input("DAP Applied (Kg/Ha)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
potash = st.sidebar.number_input("Potash Applied (Kg/Ha)", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)

# -----------------------
# Categorical Inputs
# -----------------------
# Use the loaded classes for the selectboxes
crop_choice = st.sidebar.selectbox("Crop", options=list(label_encoders['Crop'].classes_))
season_choice = st.sidebar.selectbox("Season", options=list(label_encoders['Season'].classes_))
district_choice = st.sidebar.selectbox("District", options=list(label_encoders['District'].classes_))
soil_type_choice = st.sidebar.selectbox("Soil Type", options=list(label_encoders['Soil_Type'].classes_))
irrigation_choice = st.sidebar.selectbox("Irrigation Type", options=list(label_encoders['Irrigation_Type'].classes_))

# -----------------------
# Build user input dictionary
# -----------------------
user_inputs = {
    # Numeric Features
    "Year": year,
    "Area_Hectares": area_hectares,
    "Rainfall_mm": rainfall,
    "Min_Temperature_C": min_temp,
    "Max_Temperature_C": max_temp,
    "Temp_Range": temp_range,
    "Humidity_Percent": humidity,
    "Soil_pH": soil_ph,
    "Organic_Carbon_Percent": organic_carbon,
    "NPK_Total": npk_total,
    "Fertilizer_Total": fert_total,
    "Nitrogen_KgPerHa": nitrogen,
    "Phosphorus_KgPerHa": phosphorus,
    "Potassium_KgPerHa": potassium,
    "Pest_Incidence_Score": pest_score,
    "Urea_Applied_KgPerHa": urea,
    "DAP_Applied_KgPerHa": dap,
    "Potash_Applied_KgPerHa": potash,
    # Encoded Categorical Features
    # NOTE: The names here MUST match the feature names in FEATURES list
    "Crop_Encoded": encode_category("Crop", crop_choice),
    "Season_Encoded": encode_category("Season", season_choice),
    "District_Encoded": encode_category("District", district_choice),
    "Soil_Type_Encoded": encode_category("Soil_Type", soil_type_choice),
    "Irrigation_Type_Encoded": encode_category("Irrigation_Type", irrigation_choice),
}

# -----------------------
# Preprocessing Input Data
# -----------------------
input_df = make_input_df(user_inputs)
# Convert all columns to numeric safely (though the inputs should be fine)
for col in input_df.columns:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
input_df = input_df.fillna(0) # Should handle any NaN introduced by 'coerce'

# Preview inputs
st.subheader("📋 Input Summary")
st.write(input_df.T.rename(columns={0: "Value"}))

# -----------------------
# ⭐️ Model Prediction Logic (THE MISSING PART) ⭐️
# -----------------------
if st.button("🚀 **Predict Yield**"):
    try:
        # 1. Make the prediction
        # LightGBM's predict method expects an array-like or DataFrame
        prediction_array = model.predict(input_df)
        predicted_yield = prediction_array[0] # Get the single prediction value
        
        # 2. Display the result
        # Ensure yield is not negative (possible with tree-based regression)
        display_yield = max(0, predicted_yield)
        
        st.success("✅ **Prediction Complete!**")
        st.metric(
            label="ESTIMATED YIELD (kg/ha)", 
            value=f"{display_yield:,.2f} kg/ha"
        )
        
        if display_yield == 0:
            st.warning("⚠️ The model predicted a yield of 0. This might indicate an issue with the input parameters.")
        elif display_yield < 100:
             st.info("The predicted yield is very low. Please check the environmental factors.")


    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# -----------------------
# Feature Importance
# -----------------------
st.markdown("---")
if st.checkbox("📊 Show Model Feature Importances"):
    try:
        import matplotlib.pyplot as plt
        
        # Ensure the model object has a feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_names = FEATURES
            fi = pd.Series(importances, index=feat_names).sort_values(ascending=True)
            
            # Use columns to make the plot layout cleaner
            col1, col2 = st.columns([1, 2])
            with col2:
                fig, ax = plt.subplots(figsize=(8, len(feat_names) * 0.4 + 1))
                fi.plot(kind="barh", ax=ax)
                plt.title("Feature Importances (LightGBM)", fontsize=14)
                plt.xlabel("Importance Score")
                plt.ylabel("Feature")
                plt.tight_layout()
                st.pyplot(fig)
            with col1:
                st.write("The chart shows which input variables the model used most to make the yield prediction.")
        else:
            st.warning("The loaded model object does not have a 'feature_importances_' attribute.")
            
    except Exception as e:
        st.error(f"Could not plot feature importances: {e}")

st.caption("🔬 Model trained with LightGBM. Update files in the `models/` folder when retrained.")