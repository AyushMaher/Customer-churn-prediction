# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:/Project Building/Customer churn prediction/customer_churn_dataset-training-master.csv")

# Basic information and structure of the dataset
print("Dataset Information:")
df.info()

# Summary statistics for numerical features
print("\nSummary Statistics:")
print(df.describe())

# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display the first few rows of the dataset
print("\nFirst Few Rows of Data:")
print(df.head())

# Correlation matrix to understand feature relationships
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Categorical Features Analysis
categorical_features = ['Gender', 'Subscription Type', 'Contract Length']  # Update based on actual column names

for feature in categorical_features:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=feature, hue='Churn', data=df)
    plt.title(f'{feature} vs Churn')
    plt.show()

# Numerical Features Analysis (Distribution and Churn Comparison)
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('Churn')  # Remove target variable if in the list

for feature in numerical_features:
    plt.figure(figsize=(12, 4))

    # Distribution plot
    plt.subplot(1, 2, 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'{feature} Distribution')

    # Boxplot comparison with Churn
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Churn', y=feature, data=df)
    plt.title(f'{feature} vs Churn')
    
    plt.show()

# Churn Rate by Subscription Type and Contract Length
for feature in ['Subscription Type', 'Contract Length']:
    churn_rate = df.groupby(feature)['Churn'].mean()
    plt.figure(figsize=(8, 4))
    churn_rate.plot(kind='bar', color='skyblue')
    plt.title(f'Churn Rate by {feature}')
    plt.ylabel('Churn Rate')
    plt.show()

# Pair Plot for relationships in numerical data
sns.pairplot(df, hue='Churn', corner=True, plot_kws={'alpha': 0.5})
plt.suptitle('Pair Plot of Numerical Features', y=1.02)
plt.show()

# Summary of Insights
print("\nPotential Insights:")
print("1. Visualize correlation and feature relationships with churn.")
print("2. Analyze the impact of subscription type and contract length on churn.")
print("3. Identify key factors contributing to churn based on numerical distributions.")
