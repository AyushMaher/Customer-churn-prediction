# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:31:57 2024

@author: ayush
"""

# -*- coding: utf-8 -*-
"""
Optimized Machine Learning Model for Customer Churn Prediction with Bias Checks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
)
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("C:/Project Building/Customer churn prediction/customer_churn_dataset-training-master.csv")

# Drop 'CustomerID' from the features
df = df.drop(columns=['CustomerID'])

# Check for class imbalance in the target variable
print("Class distribution in 'Churn':")
print(df['Churn'].value_counts(normalize=True))

# Handle missing values by dropping rows with missing features
X = df.drop(columns=['Churn']).dropna()
y = df['Churn'].loc[X.index]  # Align y with the updated X

# Encode categorical variables (Gender, Subscription Type, Contract Length)
label_encoders = {}
for column in ['Gender', 'Subscription Type', 'Contract Length']:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Split data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Standardize numeric features, excluding the categorical ones
scaler = StandardScaler()
numeric_columns = X.select_dtypes(include=[np.number]).columns
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Initialize and train the Random Forest model with class weights to handle imbalance
random_forest = RandomForestClassifier(class_weight='balanced', random_state=42)
random_forest.fit(X_train, y_train)

# Stratified Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(random_forest, X_train, y_train, cv=cv)
print(f"Stratified Cross-Validation Accuracy: {cv_scores.mean():.2f}")

# Training and test accuracy
train_accuracy = random_forest.score(X_train, y_train)
test_accuracy = random_forest.score(X_test, y_test)
print(f"Random Forest Training Accuracy: {train_accuracy:.2f}")
print(f"Random Forest Test Accuracy: {test_accuracy:.2f}")

# Model predictions and evaluation metrics
y_pred = random_forest.predict(X_test)
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

# Feature Importance Analysis
feature_importances = pd.Series(random_forest.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:\n", feature_importances)

# Save the model and scaler
import pickle
model_filename = "new_model.pickle"
with open(model_filename, 'wb') as file:
    pickle.dump(random_forest, file)
print(f"Model saved as {model_filename}")

scaler_filename = "new_scaler.pickle"
with open(scaler_filename, "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)
print(f"Scaler saved as {scaler_filename}")

encoders_filename = "new_label_encoders.pickle"
with open(encoders_filename, "wb") as encoders_file:
    pickle.dump(label_encoders, encoders_file)
print(f"Label encoders saved as {encoders_filename}")
