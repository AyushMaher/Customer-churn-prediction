


import pickle

# Save the scaler
with open(r"C:/Project Building/Customer churn prediction/scaler.pickle", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save the label encoders
with open(r"C:/Project Building/Customer churn prediction/label_encoders.pickle", "wb") as encoders_file:
    pickle.dump(label_encoders, encoders_file)



