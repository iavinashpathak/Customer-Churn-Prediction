import predictor

# Create a new customer
new_customer = {
    'gender': 'Male',
    'SeniorCitizen': 1,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 90.0,
    'TotalCharges': 90.0
}

# Get the prediction
prediction_result = predictor.predict_churn(new_customer)

print("--- Prediction from app.py ---")
print(f"Will this customer churn? {prediction_result['churn']}")
print(f"Probability of churn: {prediction_result['probability_yes']:.2%}")