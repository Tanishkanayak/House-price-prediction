import joblib
import numpy as np
import streamlit as st

# Load the saved model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏡 House Price Prediction App")

# Input fields
amount = st.number_input("Amount (in rupees)", min_value=0)
carpet_area = st.number_input("Carpet Area (sq ft)", min_value=0)
floor = st.number_input("Floor", min_value=0)
bathroom = st.number_input("Number of Bathrooms", min_value=0)
balcony = st.number_input("Number of Balconies", min_value=0)
car_parking = st.number_input("Car Parking", min_value=0)
super_area = st.number_input("Super Area (sq ft)", min_value=0)

# Predict button
if st.button("Predict Price"):
    features = np.array([amount, carpet_area, floor, bathroom, balcony, car_parking, super_area]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    predicted_price = model.predict(features_scaled)
    st.success(f"🏠 Estimated House Price: ₹{predicted_price[0]:,.2f}")
    