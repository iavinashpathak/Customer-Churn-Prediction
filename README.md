End-to-End Customer Churn Prediction 🚀

This project is a complete end-to-end data science workflow, predicting telecom customer churn from a raw CSV to a deployed, interactive web application.

Live Demo

The final model is deployed on Streamlit Cloud.

➡️ Try the live Customer Churn Predictor!

(This is a placeholder GIF - you should record your own!)

⚙️ Project Workflow

The entire process followed the standard machine learning lifecycle:

1. Data Cleaning & EDA (Jupyter Notebook)

Loaded the WA_Fn-UseC_-Telco-Customer-Churn.csv dataset.

Handled problematic data types (e.g., TotalCharges was an object type).

Corrected categorical values (e.g., mapping 'No internet service' to 'No').

Analyzed feature relationships with the target variable (Churn).

2. Preprocessing & Feature Engineering

Built a robust ColumnTransformer to create a single preprocessing pipeline.

Numeric Features (tenure, MonthlyCharges, TotalCharges): Imputed missing values (NaNs) with SimpleImputer and scaled with StandardScaler.

Categorical Features: Encoded all categorical data using OneHotEncoder.

3. Model Training & Tuning

Established a baseline with LogisticRegression.

Trained DecisionTree and RandomForest models, identifying clear overfitting (99% train score vs. ~75% test score).

Best Model (XGBoost): Trained an XGBClassifier which provided the best balance of performance and generalization.

Hyperparameter Tuning: Used GridSearchCV to find the optimal hyperparameters for the XGBoost model, focusing on max_depth, learning_rate, and n_estimators to reduce overfitting.

Final Model: The best-tuned XGBoost model pipeline (preprocessor + model) was saved as churn_predictor_pipeline.joblib.

4. Deployment

Backend (predictor.py): A simple Python script that loads the .joblib pipeline and contains a single function to predict new data.

Frontend (app_streamlit.py): An interactive web app built with Streamlit. It provides a user-friendly form to input customer data.

Cloud (Streamlit Cloud): The app is connected to this GitHub repo and deployed on Streamlit's free cloud platform.

🔧 How to Run Locally

Clone this repository:

git clone [https://github.com/iavinashpathak/Customer-Churn-Prediction.git](https://github.com/iavinashpathak/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction


Install the required libraries:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app_streamlit.py


Your browser will automatically open to the app.

Technologies Used

Data Analysis: Pandas, NumPy, Matplotlib, Seaborn

ML Pipeline: Scikit-learn (Pipeline, ColumnTransformer, GridSearchCV)

Modeling: XGBoost, Scikit-learn (LogisticRegression, RandomForestClassifier)

Deployment: Streamlit

Model Serving: Joblib