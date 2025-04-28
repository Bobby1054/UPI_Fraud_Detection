# predict.py

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('random_forest_upi_fraud_model.pkl')

# Example new transaction (you can change the values)
new_transaction = pd.DataFrame({
    'type': ['TRANSFER'],  # Try changing to 'CASH_OUT', 'PAYMENT', etc.
    'amount': [5000],
    'oldbalanceOrg': [10000],
    'newbalanceOrig': [5000]
})

# Encode the 'type' column same way as during training
le = LabelEncoder()
new_transaction['type'] = le.fit_transform(new_transaction['type'])

# Predict
prediction = model.predict(new_transaction)

# Show the result
if prediction[0] == 1:
    print("⚠️  Alert: Fraudulent Transaction Detected!")
else:
    print("✅ Safe: Transaction Looks Legitimate.")
