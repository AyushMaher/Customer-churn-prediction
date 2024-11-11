import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load Model
def load_model():
    with open('random_forest_churn_model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

# Sidebar - Upload Dataset
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load and display dataset
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

if uploaded_file:
    data = load_data(uploaded_file)
    st.title("Customer Churn Analysis")
    
    # Display dataset information
    st.header("Dataset Overview")
    if st.checkbox("Show Dataset"):
        st.write(data)
    st.write("Number of rows:", data.shape[0])
    st.write("Number of columns:", data.shape[1])
    
    # Display summary statistics
    if st.checkbox("Show Summary Statistics"):
        st.write(data.describe())
    
    # Correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Display relationship with Churn
    st.header("Churn Analysis")
    if st.checkbox("Show Churn Distribution"):
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Churn", data=data, ax=ax)
        st.pyplot(fig)

    # Feature-based analysis
    if st.checkbox("Analyze Features"):
        st.subheader("Feature Analysis")
        col_type = st.selectbox("Choose feature type", ["Numerical", "Categorical"])
        if col_type == "Numerical":
            numerical_cols = data.select_dtypes(include=np.number).columns
            feature = st.selectbox("Select a numerical feature", numerical_cols)
            fig, ax = plt.subplots()
            sns.histplot(data[feature], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            categorical_cols = data.select_dtypes(include='object').columns
            feature = st.selectbox("Select a categorical feature", categorical_cols)
            fig, ax = plt.subplots()
            sns.countplot(x=feature, data=data, ax=ax)
            st.pyplot(fig)

    # Customer Churn Prediction Form
    st.sidebar.header("Customer Churn Prediction")
    model = load_model()
    st.sidebar.write("Fill in customer details to predict churn.")

    # Prediction form
    def get_prediction(input_data):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        return "Churn" if prediction == 1 else "No Churn"

    with st.sidebar.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        income = st.number_input("Income", min_value=1000, max_value=1000000, step=100)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, step=1)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, step=0.1)
        
        # Collect input data
        input_data = {
            "age": age,
            "income": income,
            "marital_status": marital_status,
            "dependents": dependents,
            "tenure": tenure,
            "monthly_charges": monthly_charges
        }

        if st.form_submit_button("Predict"):
            prediction = get_prediction(input_data)
            st.sidebar.write(f"Prediction: {prediction}")

else:
    st.write("Please upload a dataset to proceed.")
