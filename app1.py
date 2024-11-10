import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Customer Churn Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display dataset information
    st.subheader("Dataset Information")
    buffer = df.info(buf=None)
    st.text(buffer)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Display first few rows of the dataset
    st.subheader("First Few Rows of Data")
    st.write(df.head())
    
    # Filter numeric columns for correlation heatmap
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Correlation heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Target Variable Distribution
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax)
    st.pyplot(fig)

    # Categorical Features Analysis
    categorical_features = ['Gender', 'Subscription Type', 'Contract Length']  # Modify as needed
    for feature in categorical_features:
        st.subheader(f"{feature} vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(x=feature, hue='Churn', data=df, ax=ax)
        st.pyplot(fig)

    # Numerical Features Analysis
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in numerical_features:
        numerical_features.remove('Churn')  # Exclude target variable

    for feature in numerical_features:
        st.subheader(f"{feature} Analysis")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Distribution plot
        sns.histplot(df[feature], kde=True, ax=ax1)
        ax1.set_title(f'{feature} Distribution')
        
        # Boxplot comparison with Churn
        sns.boxplot(x='Churn', y=feature, data=df, ax=ax2)
        ax2.set_title(f'{feature} vs Churn')
        
        st.pyplot(fig)

    # Churn Rate by Subscription Type and Contract Length
    for feature in ['Subscription Type', 'Contract Length']:  # Modify if needed
        churn_rate = df.groupby(feature)['Churn'].mean()
        st.subheader(f"Churn Rate by {feature}")
        fig, ax = plt.subplots()
        churn_rate.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_ylabel('Churn Rate')
        st.pyplot(fig)

    # Pair Plot for relationships in numerical data
    st.subheader("Pair Plot of Numerical Features")
    fig = sns.pairplot(df, hue='Churn', corner=True, plot_kws={'alpha': 0.5})
    st.pyplot(fig)

    # Summary of Insights
    st.subheader("Insights Summary")
    st.write("""
    - Visualize correlations and relationships between features with churn.
    - Analyze impact of subscription type and contract length on churn.
    - Identify key factors contributing to churn based on numerical distributions.
    """)
