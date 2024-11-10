# -*- coding: utf-8 -*-
"""
Optimized Machine Learning Model for Customer Churn Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, mean_squared_error, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("C:/Project Building/Customer churn prediction/customer_churn_dataset-training-master.csv")

# Drop 'CustomerID' from the features
df = df.drop(columns=['CustomerID'])

# Check and handle missing values
X = df.drop(columns=['Churn']).dropna()  # Dropping rows with missing values in features
y = df['Churn'].loc[X.index]  # Align y with the updated X
df.columns
# Encode categorical variables (Gender, Subscription Type, Contract Length)
label_encoders = {}
for column in ['Gender', 'Subscription Type', 'Contract Length']:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Random Forest model
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(random_forest, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")

# Training and test accuracy
train_accuracy = random_forest.score(X_train, y_train)
test_accuracy = random_forest.score(X_test, y_test)
print(f"Random Forest Training Accuracy: {train_accuracy:.2f}")
print(f"Random Forest Test Accuracy: {test_accuracy:.2f}")

# Model predictions
y_pred = random_forest.predict(X_test)

# Evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Specificity calculation
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

# Display results
print("Confusion Matrix:\n", conf_matrix)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
import pickle
model_filename = "random_forest_churn_model.pickle"
with open(model_filename, 'wb') as file:
    pickle.dump(random_forest, file)
print(f"Model saved as {model_filename}")





import pickle

# Save the scaler
with open(r"C:/Project Building/Customer churn prediction/scaler.pickle", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save the label encoders
with open(r"C:/Project Building/Customer churn prediction/label_encoders.pickle", "wb") as encoders_file:
    pickle.dump(label_encoders, encoders_file)



non_churn_data = df[df['Churn'] == 0]
non_churn_stats = non_churn_data.describe()
print(non_churn_stats)
