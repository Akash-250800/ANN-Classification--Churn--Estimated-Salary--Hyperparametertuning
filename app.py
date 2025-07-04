import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Load model and encoders
model = tf.keras.models.load_model("regression_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title(" Customer Churn Prediction")
st.markdown("Enter customer details to predict churn probability.")

# Inputs
geography = st.sidebar.selectbox(" Geography", onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox(" Gender", label_encoder_gender.classes_)
age = st.sidebar.slider(" Age", 18, 92, 35)
creditscore = st.sidebar.number_input(" Credit Score", min_value=300, max_value=850, value=600)
tenure = st.sidebar.slider(" Tenure (Years)", 0, 10, 3)
balance = st.sidebar.number_input(" Account Balance", min_value=0.0, value=50000.0)
num_products = st.sidebar.selectbox(" Number of Products", [1, 2, 3, 4])
has_card = st.sidebar.selectbox(" Has Credit Card", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", min_value=10000.0, value=50000.0)

if st.button("Predict Churn"):
    df_input = pd.DataFrame([{
        'CreditScore': creditscore,
        'Gender': label_encoder_gender.transform([gender])[0],
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': salary
    }])

    # One-hot encode geography
    geo_encoded = onehot_encoder_geo.transform([[geography]])
    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    df_combined = pd.concat([df_input, geo_df], axis=1)

    expected_columns = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_France', 'Geography_Germany', 'Geography_Spain'
    ]

    for col in expected_columns:
        if col not in df_combined.columns:
            df_combined[col] = 0

    df_combined = df_combined[expected_columns]

    input_scaled = scaler.transform(df_combined)

    raw_output = model.predict(input_scaled)[0][0]
    churn_probability = 1 / (1 + np.exp(-raw_output))  # sigmoid

    st.write(f"🔍 Raw model output: {raw_output:.4f}")
    st.write(f"🔢 **Churn Probability:** {churn_probability*100:.2f}%")

    if churn_probability > 0.6:
        st.warning(" The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
