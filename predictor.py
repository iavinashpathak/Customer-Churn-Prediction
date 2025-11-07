import joblib
import pandas as pd
import numpy as np

# --- 1. Load the "brain" (our saved pipeline) ---
MODEL_PATH = 'churn_predictor_pipeline.joblib'

try:
    # Load the pipeline object from the file
    MODEL_PIPELINE = joblib.load(MODEL_PATH)
    print("Model pipeline loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    MODEL_PIPELINE = None
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_PIPELINE = None

def predict_churn(raw_data_dict):
    """
    Predicts churn for a single customer provided as a dictionary.

    Args:
        raw_data_dict (dict): A dictionary with keys matching the
                              features used in the training data 
                              (e.g., 'tenure', 'gender', 'MonthlyCharges').

    Returns:
        dict: A dictionary containing the prediction ('Yes'/'No')
              and the churn probability.
              e.g., {'churn': 'No', 'probability_yes': 0.123, 'status': 'OK'}
    """
    if MODEL_PIPELINE is None:
        return {"error": "Model is not loaded.", "status": "Error"}

    try:
        # --- 2. Convert the single dictionary into a DataFrame ---
        # The ColumnTransformer in our pipeline expects a DataFrame
        # We use [raw_data_dict] to create a DataFrame with one row
        df = pd.DataFrame([raw_data_dict])

        # --- 3. Get the prediction (0 or 1) ---
        # .predict() gives the final class
        # [0] extracts the prediction for the first (and only) row
        prediction_class = MODEL_PIPELINE.predict(df)[0]

        # --- 4. Get the probabilities ---
        # .predict_proba() gives [prob_of_0, prob_of_1]
        # [0] gets the probabilities for the first row
        probabilities = MODEL_PIPELINE.predict_proba(df)[0]
        
        # [1] gets the probability of class 1 ('Yes')
        probability_yes = probabilities[1] 

        # --- 5. Format the output ---
        churn_label = 'Yes' if prediction_class == 1 else 'No'

        return {
            'churn': churn_label,
            'prediction_class': int(prediction_class),
            'probability_yes': float(probability_yes),
            'status': 'OK'
        }
    
    except Exception as e:
        # Catch any errors during transformation or prediction
        return {"error": f"Error during prediction: {e}", "status": "Error"}

# --- 6. Example Usage (for testing the script directly) ---
if __name__ == '__main__':
    """
    This block runs only when you execute this script directly
    (e.g., by running 'python predictor.py' in your terminal)
    """
    print("\n--- Testing predictor.py ---")
    
    # Create a sample customer who is likely to churn
    sample_customer_churn = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.70,
        'TotalCharges': 151.65
    }
    
    # Get prediction
    result = predict_churn(sample_customer_churn)
    print("\nPrediction for sample (high-risk) customer:")
    print(result)

    # Create a sample customer who is likely to stay
    sample_customer_loyal = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 70,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Credit card (automatic)',
        'MonthlyCharges': 80.00,
        'TotalCharges': 5600.00
    }

    result_loyal = predict_churn(sample_customer_loyal)
    print("\nPrediction for sample (loyal) customer:")
    print(result_loyal)