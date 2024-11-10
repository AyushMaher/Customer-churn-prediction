# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model, scaler, and encoders
with open("C:/Project Building/Customer churn prediction/random_forest_churn_model.pickle", "rb") as model_file:
    model = pickle.load(model_file)

with open("C:/Project Building/Customer churn prediction/scaler.pickle", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("C:/Project Building/Customer churn prediction/label_encoders.pickle", "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)

# Define Streamlit app
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if the customer will churn or not.")

# Collect input from the user
age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, step=1)
usage_frequency = st.number_input("Usage Frequency", min_value=1, max_value=10, step=1)
support_calls = st.number_input("Support Calls", min_value=0, max_value=20, step=1)
payment_delay = st.number_input("Payment Delay (months)", min_value=0, max_value=12, step=1)
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.selectbox("Contract Length", ["Month-to-Month", "One Year", "Two Year"])
total_spend = st.number_input("Total Spend", min_value=0.0, max_value=5000.0, step=1.0)
last_interaction = st.number_input("Last Interaction (months)", min_value=0, max_value=72, step=1)

# Encode categorical inputs using the label encoders
gender_encoded = label_encoders["Gender"].transform([gender])[0]
subscription_encoded = label_encoders["Subscription Type"].transform([subscription_type])[0]
contract_encoded = label_encoders["Contract Length"].transform([contract_length])[0]

# Prepare the data for prediction
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_encoded],
    "Tenure": [tenure],
    "Usage Frequency": [usage_frequency],
    "Support Calls": [support_calls],
    "Payment Delay": [payment_delay],
    "Subscription Type": [subscription_encoded],
    "Contract Length": [contract_encoded],
    "Total Spend": [total_spend],
    "Last Interaction": [last_interaction]
})

# Scale numeric fields
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
churn_prob = model.predict_proba(input_data_scaled)[0][1]

# Display result
if prediction[0] == 1:
    st.write("Prediction: This customer is likely to **Churn**.")
else:
    st.write("Prediction: This customer is **Not Likely to Churn**.")

st.write(f"Churn Probability: {churn_prob:.2f}")
