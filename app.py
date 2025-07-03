import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model.h5")

# Load encoders and scaler
with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------
# Streamlit UI inputs
# ----------------------
st.title("Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92, 35)
creditscore = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", min_value=10000.0, value=50000.0)

# ----------------------
# Prepare input DataFrame
# ----------------------
input_data = pd.DataFrame([{
    "CreditScore": creditscore,
    "Gender": label_encoder_gender.transform([gender])[0],  # LABEL encoded
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary
}])

# One-hot encode geography (only geography)
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))

# Combine with input
df_combined = pd.concat([input_data, geo_df], axis=1)

# Expected column order
expected_columns = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]

# Add missing columns if any
for col in expected_columns:
    if col not in df_combined.columns:
        df_combined[col] = 0

df_combined = df_combined[expected_columns]

# ----------------------
# Scale and Predict
# ----------------------
input_scaled = scaler.transform(df_combined)
prediction = model.predict(input_scaled)
churn_prob = prediction[0][0]

st.write(f"Churn Probability: **{churn_prob:.2f}**")
if churn_prob > 0.5:
    st.error("The customer is likely to churn.")
else:
    st.success("The customer is not likely to churn.")
