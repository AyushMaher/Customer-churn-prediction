import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load your pre-trained model (replace 'your_model.pkl' with the model file)
with open('random_forest_churn_model.pickle', 'rb') as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction")

# Create input fields for each feature in the dataset
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
usage_frequency = st.number_input("Usage Frequency", min_value=0, max_value=50, value=10)
support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=5)
payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=30, value=0)
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0)
last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=365, value=30)

# Encode categorical features as needed for the model
gender_encoded = 1 if gender == "Female" else 0
subscription_mapping = {"Basic": 0, "Standard": 1, "Premium": 2}
contract_mapping = {"Monthly": 0, "Quarterly": 1, "Annual": 2}
subscription_encoded = subscription_mapping[subscription_type]
contract_encoded = contract_mapping[contract_length]

# Prepare the input data for prediction
input_data = np.array([[age, gender_encoded, tenure, usage_frequency, support_calls, payment_delay,
                        subscription_encoded, contract_encoded, total_spend, last_interaction]])

# Make a prediction when the button is clicked
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    churn_status = "will churn" if prediction[0] == 1 else "will not churn"
    st.write(f"The model predicts that this customer {churn_status}.")
