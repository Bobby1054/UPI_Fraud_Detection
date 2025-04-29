# main.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('PS_20174392719_1491204439457_log.csv')

# Create a new feature: errorBalanceOrig
data['errorBalanceOrig'] = data['oldbalanceOrg'] - data['amount'] - data['newbalanceOrig']

# Select relevant columns for features (X) and label (y)
X = data[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrig']].copy()
y = data['isFraud']

# Encode the 'type' column
le = LabelEncoder()
X['type'] = le.fit_transform(X['type'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Predict on the test set
y_pred = rfc.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy of the Random Forest model: {accuracy*100:.2f}%")

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and the label encoder together
joblib.dump((rfc, le), 'random_forest_upi_fraud_model.pkl')
print("\n Model and encoder saved successfully as 'random_forest_upi_fraud_model.pkl'")
