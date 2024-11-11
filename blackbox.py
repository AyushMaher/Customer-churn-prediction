import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the pre-trained model
with open('random_forest_churn_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Streamlit App Title
st.title("Customer Churn Analysis and Prediction")

# File Upload Section
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset Overview Section
    st.header("1. Dataset Overview")
    if st.checkbox("Show dataset information"):
        buffer = df.info(buf=None)
        st.text(buffer)

    if st.checkbox("Show summary statistics"):
        st.write(df.describe())

    if st.checkbox("Show missing values"):
        st.write(df.isnull().sum())

    if st.checkbox("Show first few rows"):
        st.write(df.head())

    # # Correlation Analysis
    # st.header("2. Correlation Analysis")
    # if st.checkbox("Show correlation matrix"):
    #     plt.figure(figsize=(12, 8))
    #     sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    #     st.pyplot(plt.gcf())
    #     plt.clf()

    # Target Variable Distribution
    st.header("2. Target Variable (Churn) Distribution")
    if st.checkbox("Show churn distribution"):
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Churn', data=df)
        st.pyplot(plt.gcf())
        plt.clf()

    # Categorical Features Analysis
    st.header("3. Categorical Features Analysis")
    categorical_features = ['Gender', 'Subscription Type', 'Contract Length']
    for feature in categorical_features:
        if st.checkbox(f"Show {feature} vs Churn"):
            plt.figure(figsize=(6, 4))
            sns.countplot(x=feature, hue='Churn', data=df)
            st.pyplot(plt.gcf())
            plt.clf()

    # Numerical Features Analysis
    st.header("4. Numerical Features Analysis")
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('Churn')

    for feature in numerical_features:
        if st.checkbox(f"Show distribution and Churn comparison for {feature}"):
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            sns.histplot(df[feature], kde=True)
            plt.title(f'{feature} Distribution')

            plt.subplot(1, 2, 2)
            sns.boxplot(x='Churn', y=feature, data=df)
            plt.title(f'{feature} vs Churn')

            st.pyplot(plt.gcf())
            plt.clf()

    # Pair Plot
    st.header("5. Pair Plot for Numerical Features")
    if st.checkbox("Show pair plot"):
        sns.pairplot(df, hue='Churn', corner=True, plot_kws={'alpha': 0.5})
        st.pyplot(plt.gcf())
        plt.clf()

# Customer Churn Prediction Section
st.header("Customer Churn Prediction")

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

# Encode categorical features
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
