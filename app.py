import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('ridge_car_price_model.pkl')

st.title("🚗 Second-Hand Car Price Predictor (OLX Pakistan Style)")
st.write("Enter your car's details to estimate its resale price.")

# Input fields
brand = st.selectbox("Brand", ["Suzuki", "Toyota", "Honda", "KIA", "Hyundai", "Changan"])
city = st.selectbox("City", ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Sahiwal", "Multan"])
age_years = st.slider("Car Age (years)", 0, 20, 5)
mileage_km = st.number_input("Mileage (km)", min_value=0, max_value=300000, value=50000, step=1000)
engine_cc = st.selectbox("Engine CC", [660, 1000, 1300, 1500, 1800])
condition_score = st.slider("Condition Score (1-10)", 1.0, 10.0, 7.0)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'brand': [brand],
        'city': [city],
        'age_years': [age_years],
        'mileage_km': [mileage_km],
        'engine_cc': [engine_cc],
        'condition_score': [condition_score]
    })
    
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: PKR {prediction:,.0f}")