# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("C:/Project Building/Customer churn prediction/customer_churn_dataset-training-master.csv")

df = load_data()

# Streamlit App
st.title("Customer Churn Analysis")

# Dataset Overview Section
st.header("1. Dataset Overview")
if st.checkbox("Show dataset information"):
    st.write("Dataset Information:")
    buffer = st.text(df.info())
    st.text(buffer)

if st.checkbox("Show summary statistics"):
    st.write("Summary Statistics")
    st.write(df.describe())

if st.checkbox("Show missing values"):
    st.write("Missing Values")
    st.write(df.isnull().sum())

if st.checkbox("Show first few rows"):
    st.write("First Few Rows of Data")
    st.write(df.head())

# Correlation Analysis
st.header("2. Correlation Analysis")
if st.checkbox("Show correlation matrix"):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf())
    plt.clf()

# Target Variable Distribution
st.header("3. Target Variable (Churn) Distribution")
if st.checkbox("Show churn distribution"):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    st.pyplot(plt.gcf())
    plt.clf()

# Categorical Features Analysis
st.header("4. Categorical Features Analysis")
categorical_features = ['Gender', 'Subscription Type', 'Contract Length']  # Update based on actual column names

for feature in categorical_features:
    if st.checkbox(f"Show {feature} vs Churn"):
        plt.figure(figsize=(6, 4))
        sns.countplot(x=feature, hue='Churn', data=df)
        plt.title(f'{feature} vs Churn')
        st.pyplot(plt.gcf())
        plt.clf()

# Numerical Features Analysis
st.header("5. Numerical Features Analysis")
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('Churn')  # Remove target variable if in the list

for feature in numerical_features:
    if st.checkbox(f"Show distribution and Churn comparison for {feature}"):
        plt.figure(figsize=(12, 4))

        # Distribution plot
        plt.subplot(1, 2, 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'{feature} Distribution')

        # Boxplot comparison with Churn
        plt.subplot(1, 2, 2)
        sns.boxplot(x='Churn', y=feature, data=df)
        plt.title(f'{feature} vs Churn')

        st.pyplot(plt.gcf())
        plt.clf()

# Churn Rate by Subscription Type and Contract Length
st.header("6. Churn Rate by Categorical Features")
for feature in ['Subscription Type', 'Contract Length']:
    if st.checkbox(f"Show churn rate by {feature}"):
        churn_rate = df.groupby(feature)['Churn'].mean()
        plt.figure(figsize=(8, 4))
        churn_rate.plot(kind='bar', color='skyblue')
        plt.title(f'Churn Rate by {feature}')
        plt.ylabel('Churn Rate')
        st.pyplot(plt.gcf())
        plt.clf()

# Pair Plot
st.header("7. Pair Plot for Numerical Features")
if st.checkbox("Show pair plot"):
    sns.pairplot(df, hue='Churn', corner=True, plot_kws={'alpha': 0.5})
    st.pyplot(plt.gcf())

# Summary of Insights
st.header("8. Summary of Insights")
st.write("""
- Correlation matrix helps in understanding feature relationships.
- Distribution analysis reveals patterns and trends for churned customers.
- Visualize categorical features like subscription type and contract length with churn.
""")
