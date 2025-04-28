import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('PS_20174392719_1491204439457_log.csv')

# Select relevant columns
X = data[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
y = data['isFraud']

# Encode the 'type' column
le = LabelEncoder()
X['type'] = le.fit_transform(X['type'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Predict on the test set
y_pred = rfc.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest model: {accuracy*100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# Save the trained model
joblib.dump(rfc, 'random_forest_upi_fraud_model.pkl')
print("\nModel saved successfully as 'random_forest_upi_fraud_model.pkl'")
