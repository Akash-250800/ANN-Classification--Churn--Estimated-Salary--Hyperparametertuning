import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load the trained model and scaler
model = load_model("model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit setup
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title(" Customer Churn Prediction App")
st.markdown("Please enter customer details below:")

# User inputs
credit_score = st.slider("Credit Score", 300, 900, 600)
geo = st.radio("Geography", ["France", "Germany", "Spain"])
gender = st.radio("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Balance", value=15000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active = st.selectbox("Is Active Member", ["Yes", "No"])

# Predict button
if st.button(" Predict"):
    try:
        # Geography one-hot encoding
        geo_vals = {
            "France": [1, 0, 0],
            "Germany": [0, 1, 0],
            "Spain": [0, 0, 1]
        }[geo]

        # Gender one-hot encoding
        gender_female = 1 if gender == "Female" else 0
        gender_male = 1 if gender == "Male" else 0

        # Build input DataFrame in the correct feature order
        input_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography_France": geo_vals[0],
            "Geography_Germany": geo_vals[1],
            "Geography_Spain": geo_vals[2],
            "Gender_Female": gender_female,
            "Gender_Male": gender_male,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active == "Yes" else 0
        }])

        # Scale and predict
        scaled_input = scaler.transform(input_df)
        probability = model.predict(scaled_input)[0][0]
        prediction = " LIKELY TO CHURN" if probability > 0.5 else " NOT LIKELY TO CHURN"

        # Display result
        st.markdown(f"### Prediction: **{prediction}**")
        st.markdown(f"**Confidence Score:** `{probability:.2f}`")
        st.progress(min(float(probability), 1.0))

    except Exception as e:
        st.error(f" Error during prediction: {e}")
