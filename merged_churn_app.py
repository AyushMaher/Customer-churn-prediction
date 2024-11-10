
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load pre-trained model for predictions
with open('random_forest_churn_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Set up Streamlit app layout
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Analysis", "Data Visualization", "Churn Prediction"])

# Function to load dataset
@st.cache_data
def load_data():
    # Path may need to be modified to access the data correctly
    return pd.read_csv("C:/Project Building/Customer churn prediction/customer_churn_dataset-training-master.csv")

# Data Analysis Page
if page == "Data Analysis":
    st.title("Customer Churn Data Analysis")
    df = load_data()
    
    # Dataset Overview
    st.header("Dataset Overview")
    if st.checkbox("Show dataset information"):
        buffer = df.info()
        st.text(buffer)

    if st.checkbox("Show summary statistics"):
        st.write(df.describe())

    if st.checkbox("Show missing values"):
        st.write(df.isnull().sum())

    if st.checkbox("Show first few rows"):
        st.write(df.head())

# Data Visualization Page
elif page == "Data Visualization":
    st.title("Customer Churn Data Visualization")
    df = load_data()

    # Correlation Matrix
    st.header("Correlation Analysis")
    if st.checkbox("Show correlation matrix"):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())

    # Target Variable Distribution
    st.header("Target Variable Distribution")
    if st.checkbox("Show churn distribution"):
        plt.figure(figsize=(6, 4))
        sns.countplot(x="Churn", data=df)
        st.pyplot(plt.gcf())

    # Categorical Features Analysis
    categorical_features = ['Gender', 'Subscription Type', 'Contract Length']
    for feature in categorical_features:
        if st.checkbox(f"Show {feature} vs Churn"):
            plt.figure(figsize=(6, 4))
            sns.countplot(x=feature, hue="Churn", data=df)
            st.pyplot(plt.gcf())

    # Numerical Features Analysis
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Churn" in numerical_features:
        numerical_features.remove("Churn")

    for feature in numerical_features:
        if st.checkbox(f"Show distribution and Churn comparison for {feature}"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[feature], kde=True, ax=ax1)
            sns.boxplot(x="Churn", y=feature, data=df, ax=ax2)
            st.pyplot(fig)

# Churn Prediction Page
elif page == "Churn Prediction":
    st.title("Customer Churn Prediction")

    # User Inputs
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

    # Encoding categorical features
    gender_encoded = 1 if gender == "Female" else 0
    subscription_mapping = {"Basic": 0, "Standard": 1, "Premium": 2}
    contract_mapping = {"Monthly": 0, "Quarterly": 1, "Annual": 2}
    subscription_encoded = subscription_mapping[subscription_type]
    contract_encoded = contract_mapping[contract_length]

    # Prepare input for model
    input_data = np.array([[age, gender_encoded, tenure, usage_frequency, support_calls, payment_delay,
                            subscription_encoded, contract_encoded, total_spend, last_interaction]])
    
    # Prediction
    if st.button("Predict Churn"):
        prediction = model.predict(input_data)
        churn_status = "will churn" if prediction[0] == 1 else "will not churn"
        st.write(f"The model predicts that this customer {churn_status}.")
