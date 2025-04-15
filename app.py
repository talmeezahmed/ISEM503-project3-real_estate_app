import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained pipeline (preprocessing + model)
pipeline = joblib.load('real_estate_tuned_pipeline_model.pkl')

# Page title
st.title("🏠 Real Estate Price Predictor")
st.markdown("This app predicts the **log-transformed house price** using a trained model. The final price is shown in USD.")

# User Inputs
status = st.selectbox("Property Status", options=['ready for sale', 'ready to build'])  # OrdinalEncoder assumed alphabetical
bed = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=3)
bath = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2)
acre_lot = st.number_input("Lot Size (acres)", min_value=0.0, value=0.25)

city = st.text_input("City Name (e.g., Pittsburgh)")  # OrdinalEncoder will map it to -1 if unseen
sold_year = st.slider("Year of Last Sale", 2000, 2025, 2020)
sold_month = st.slider("Month of Last Sale", 1, 12, 6)
sold_age = 2025 - sold_year

is_large_house = st.checkbox("Large House? (>3000 sqft)", value=False)
is_large_lot = st.checkbox("Large Lot? (>1 acre)", value=False)

# Create DataFrame input
input_data = pd.DataFrame([{
    'status': status,
    'bed': bed,
    'bath': bath,
    'acre_lot': acre_lot,
    'city': city,
    'sold_year': sold_year,
    'sold_month': sold_month,
    'sold_age': sold_age,
    'is_large_house': int(is_large_house),
    'is_large_lot': int(is_large_lot)
}])

# Predict
if st.button("Predict Price"):
    try:
        log_price = pipeline.predict(input_data)[0]
        predicted_price = np.expm1(log_price)  # Inverse of log1p
        st.success(f"💰 Predicted House Price: **${predicted_price:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
