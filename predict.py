import pandas as pd
import joblib

# Load the saved model and label encoder
model, le = joblib.load('random_forest_upi_fraud_model.pkl')

# Take user input
transaction_type = input("Enter transaction type (TRANSFER / CASH_OUT / PAYMENT / etc): ").strip().upper()
amount = float(input("Enter transaction amount: "))
old_balance = float(input("Enter old balance of sender: "))
new_balance = float(input("Enter new balance of sender: "))

# First, manually check balance error
expected_new_balance = old_balance - amount

if abs(expected_new_balance - new_balance) > 1:  # Small margin allowed
    print("\n⚠️ Warning: Balance mismatch detected! Likely Fraudulent Transaction (Mathematical Check).")
else:
    # Create transaction DataFrame
    new_transaction = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [old_balance],
        'newbalanceOrig': [new_balance]
    })

    # Create the new feature
    new_transaction['errorBalanceOrig'] = new_transaction['oldbalanceOrg'] - new_transaction['amount'] - new_transaction['newbalanceOrig']

    # Encode the 'type' column
    new_transaction['type'] = le.transform(new_transaction['type'])

    # Predict using the model
    prediction = model.predict(new_transaction)

    # Show result
    if prediction[0] == 1:
        print("\n⚠️ Alert: Fraudulent Transaction Detected! (ML Model)")
    else:
        print("\n✅ Safe: Transaction Looks Legitimate. (ML Model)")
