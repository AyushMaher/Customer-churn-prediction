import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model, scaler, and label encoders
model = pickle.load(open("C:/Project Building/Customer churn prediction/random_forest_churn_model.pickle", "rb"))
scaler = pickle.load(open("C:/Project Building/Customer churn prediction/scaler.pickle", "rb"))
label_encoders = pickle.load(open("label_encoders.pickle", "rb"))

# List of feature columns as used during model training
feature_columns = ['Gender', 'Subscription Type', 'Contract Length', 'Other_Numeric_Features']  # Replace with actual feature names

# Define Streamlit interface
st.title("Customer Churn Prediction")

# User input fields
gender = st.selectbox("Gender", options=["Male", "Female"])
subscription_type = st.selectbox("Subscription Type", options=["Type1", "Type2", "Type3"])
contract_length = st.selectbox("Contract Length", options=["Short", "Medium", "Long"])

# Additional fields for numeric input (if needed)
# Example placeholder fields, replace with actual numeric features from your dataset
numeric_feature_1 = st.number_input("Numeric Feature 1", min_value=0.0, step=1.0)
numeric_feature_2 = st.number_input("Numeric Feature 2", min_value=0.0, step=1.0)

# Collect inputs in a dictionary
user_data = {
    'Gender': gender,
    'Subscription Type': subscription_type,
    'Contract Length': contract_length,
    'Other_Numeric_Features': [numeric_feature_1, numeric_feature_2]  # Add other numeric features
}

# Convert user inputs to DataFrame and reindex to match feature columns
user_df = pd.DataFrame([user_data])
user_df = user_df.reindex(columns=feature_columns, fill_value=0)  # Adjust fill_value if needed

# Encode categorical fields
for col in ['Gender', 'Subscription Type', 'Contract Length']:
    user_df[col] = label_encoders[col].transform(user_df[col])

# Scale numeric fields
numeric_columns = user_df.select_dtypes(include=[float, int]).columns
user_df[numeric_columns] = scaler.transform(user_df[numeric_columns])

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(user_df)
    prediction_proba = model.predict_proba(user_df)

    # Display the result
    churn_status = "Churn" if prediction[0] == 1 else "No Churn"
    churn_probability = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
    
    st.write(f"Prediction: **{churn_status}**")
    st.write(f"Churn Probability: **{churn_probability:.2f}**")
