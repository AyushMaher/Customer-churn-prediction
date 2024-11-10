# Merging the contents into a single Streamlit application with three sections
# Using the sidebar to switch between the three functionalities


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load necessary models and encoders if required
def load_models():
    with open("random_forest_churn_model.pickle", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pickle", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("label_encoders.pickle", "rb") as encoders_file:
        label_encoders = pickle.load(encoders_file)
    return model, scaler, label_encoders

# Sidebar for navigation
st.sidebar.title("Churn Prediction App")
section = st.sidebar.radio("Select a Section", ["Customer Churn Prediction", "Customer Churn Analysis", "Alternative Churn Prediction"])

# Section 1: Customer Churn Prediction
if section == "Customer Churn Prediction":
    st.title("Customer Churn Prediction")
    st.write("Enter customer details to predict if the customer will churn or not.")
    
    model, scaler, label_encoders = load_models()
    
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
    
    # Encode categorical inputs
    gender_encoded = label_encoders["Gender"].transform([gender])[0]
    subscription_encoded = label_encoders["Subscription Type"].transform([subscription_type])[0]
    contract_encoded = label_encoders["Contract Length"].transform([contract_length])[0]
    
    # Prepare data for prediction
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
    
    # Scale numeric fields and make predictions
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    churn_prob = model.predict_proba(input_data_scaled)[0][1]
    
    # Display result
    if prediction[0] == 1:
        st.write("Prediction: This customer is likely to **Churn**.")
    else:
        st.write("Prediction: This customer is **Not Likely to Churn**.")
    st.write(f"Churn Probability: {churn_prob:.2f}")

# Section 2: Customer Churn Analysis
elif section == "Customer Churn Analysis":
    st.title("Customer Churn Analysis")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display dataset information
        st.subheader("Dataset Information")
        st.write(df.info(buf=None))
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        st.write(df.describe())
        
        # Missing values
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        # Display correlation heatmap and churn distribution
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Churn Distribution
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax)
        st.pyplot(fig)

# Section 3: Alternative Churn Prediction
elif section == "Alternative Churn Prediction":
    st.title("Alternative Customer Churn Prediction")
    
    # Load alternative model if needed
    model, _, _ = load_models()
    
    # Collect input fields
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
    
    # Encode features and predict
    gender_encoded = 1 if gender == "Female" else 0
    subscription_mapping = {"Basic": 0, "Standard": 1, "Premium": 2}
    contract_mapping = {"Monthly": 0, "Quarterly": 1, "Annual": 2}
    subscription_encoded = subscription_mapping[subscription_type]
    contract_encoded = contract_mapping[contract_length]
    
    input_data = np.array([[age, gender_encoded, tenure, usage_frequency, support_calls, payment_delay,
                            subscription_encoded, contract_encoded, total_spend, last_interaction]])
    
    # Prediction
    if st.button("Predict Churn"):
        prediction = model.predict(input_data)
        churn_status = "will churn" if prediction[0] == 1 else "will not churn"
        st.write(f"The model predicts that this customer {churn_status}.")


# Writing the merged code to a new file to simplify further operations if needed.
# output_path = "/mnt/data/main_app.py"

# output_path
