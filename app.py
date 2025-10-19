import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the saved encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title("🌍 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn based on their details.")

# ---- User Inputs ----
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, step=1)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure (Years with Bank)', 0, 10)
balance = st.number_input('Account Balance', min_value=0.0, step=100.0)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=100.0)

# ---- Data Preprocessing ----

# Encode gender (LabelEncoder)
gender_encoded = label_encoder_gender.transform([gender])[0]

# Create base DataFrame (without Geography)
input_data = pd.DataFrame({
    'Credit Score': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'Number of Products': [num_of_products],
    'Has Credit Card': [has_cr_card],
    'Is Active Member': [is_active_member],
    'Estimated Salary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]])
if hasattr(geo_encoded, "toarray"):  # handle sparse matrices
    geo_encoded = geo_encoded.toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine both DataFrames
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Align columns to training scaler’s expectation
input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale input data
input_data_scaled = scaler.transform(input_data)

# ---- Prediction ----
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# ---- Output ----
st.subheader("🔍 Prediction Result")
st.write(f"**Churn Probability:** {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.error("🚨 The customer is **likely to churn.**")
else:
    st.success("✅ The customer is **not likely to churn.**")

st.caption("Model powered by TensorFlow & Streamlit")
