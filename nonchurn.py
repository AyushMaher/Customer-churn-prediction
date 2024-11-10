# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle

# Load the saved model, scaler, and label encoders
with open("C:/Project Building/Customer churn prediction/random_forest_churn_model.pickle", "rb") as model_file:
    model = pickle.load(model_file)

with open("C:/Project Building/Customer churn prediction/scaler.pickle", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("C:/Project Building/Customer churn prediction/label_encoders.pickle", "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)

# Generate sample entries based on typical non-churn values
sample_data = pd.DataFrame({
    'Age': [30, 40, 35, 28, 45, 38, 50, 42, 33, 37],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Tenure': [24, 30, 26, 28, 32, 27, 31, 29, 25, 28],
    'Usage Frequency': [10, 15, 12, 14, 13, 11, 16, 13, 10, 12],
    'Support Calls': [1, 2, 0, 1, 1, 2, 1, 0, 1, 2],
    'Payment Delay': [5, 4, 6, 4, 5, 4, 6, 5, 4, 5],
    'Subscription Type': ['Standard', 'Basic', 'Premium', 'Standard', 'Basic', 'Premium', 'Standard', 'Basic', 'Premium', 'Standard'],
    'Contract Length': ['Annual', 'Monthly', 'Quarterly', 'Annual', 'Monthly', 'Quarterly', 'Annual', 'Monthly', 'Quarterly', 'Annual'],
    'Total Spend': [500, 700, 800, 600, 550, 750, 720, 680, 640, 690],
    'Last Interaction': [20, 15, 18, 22, 19, 17, 16, 21, 14, 20]
})

# Encode categorical variables and scale the numeric features
for column in ['Gender', 'Subscription Type', 'Contract Length']:
    sample_data[column] = label_encoders[column].transform(sample_data[column])

sample_data_scaled = scaler.transform(sample_data)

# Verify the model prediction
predictions = model.predict(sample_data_scaled)
non_churn_data = sample_data[predictions == 0]  # Select entries with non-churn prediction

# Save the non-churn entries to a CSV file
non_churn_data.to_csv("non_churn_sample_entries.csv", index=False)
print("10 sample entries where the user will not churn have been saved to 'non_churn_sample_entries.csv'.")
