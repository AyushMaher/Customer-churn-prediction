import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and resources once at the start of the app
model_path = "C:/Project Building/Customer churn prediction/"
with open(model_path + "random_forest_churn_model.pickle", "rb") as model_file:
    model = pickle.load(model_file)
with open(model_path + "scaler.pickle", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open(model_path + "label_encoders.pickle", "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)

# Set up main page
st.title("Customer Churn Application")

# Sidebar selection for different sections
section = st.sidebar.selectbox("Choose a section", ["Churn Prediction", "Churn Data Analysis", "Prediction with Custom Inputs"])

if section == "Churn Prediction":
    st.header("Customer Churn Prediction")
    st.write("Enter customer details to predict if the customer will churn or not.")

    # Input fields for prediction
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

    # Encoding categorical inputs
    gender_encoded = label_encoders["Gender"].transform([gender])[0]
    subscription_encoded = label_encoders["Subscription Type"].transform([subscription_type])[0]
    contract_encoded = label_encoders["Contract Length"].transform([contract_length])[0]

    # Prepare input data
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

    # Scale numeric data
    input_data_scaled = scaler.transform(input_data)

    # Predict churn
    if st.button("Predict"):
        prediction = model.predict(input_data_scaled)
        churn_prob = model.predict_proba(input_data_scaled)[0][1]
        if prediction[0] == 1:
            st.write("Prediction: This customer is likely to **Churn**.")
        else:
            st.write("Prediction: This customer is **Not Likely to Churn**.")
        st.write(f"Churn Probability: {churn_prob:.2f}")

elif section == "Churn Data Analysis":
    st.header("Customer Churn Analysis")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Dataset Information")
        st.write(df.info())

        st.subheader("Summary Statistics")
        st.write(df.describe())
        
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("First Few Rows of Data")
        st.write(df.head())

        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        # Correlation heatmap
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax)
        st.pyplot(fig)

        categorical_features = ['Gender', 'Subscription Type', 'Contract Length']
        for feature in categorical_features:
            st.subheader(f"{feature} vs Churn")
            fig, ax = plt.subplots()
            sns.countplot(x=feature, hue='Churn', data=df, ax=ax)
            st.pyplot(fig)

        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numerical_features:
            numerical_features.remove('Churn')

        for feature in numerical_features:
            st.subheader(f"{feature} Analysis")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[feature], kde=True, ax=ax1)
            ax1.set_title(f'{feature} Distribution')
            sns.boxplot(x='Churn', y=feature, data=df, ax=ax2)
            ax2.set_title(f'{feature} vs Churn')
            st.pyplot(fig)

        st.subheader("Insights Summary")
        st.write("""
        - Visualize correlations and relationships between features with churn.
        - Analyze impact of subscription type and contract length on churn.
        - Identify key factors contributing to churn based on numerical distributions.
        """)

elif section == "Prediction with Custom Inputs":
    st.header("Customer Churn Prediction with Custom Inputs")
    
    # Custom input fields
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

    gender_encoded = 1 if gender == "Female" else 0
    subscription_encoded = {"Basic": 0, "Standard": 1, "Premium": 2}[subscription_type]
    contract_encoded = {"Monthly": 0, "Quarterly": 1, "Annual": 2}[contract_length]

    input_data = np.array([[age, gender_encoded, tenure, usage_frequency, support_calls, payment_delay,
                            subscription_encoded, contract_encoded, total_spend, last_interaction]])

    if st.button("Predict Churn"):
        prediction = model.predict(input_data)
        churn_status = "will churn" if prediction[0] == 1 else "will not churn"
        st.write(f"The model predicts that this customer {churn_status}.")
