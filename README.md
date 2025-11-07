# 📊 End-to-End Telecom Customer Churn Prediction 🚀

This project is a complete **end-to-end data science workflow** that predicts telecom customer churn — starting from raw CSV data and ending with a **fully deployed interactive web app**.

🔗 **Live Demo:**  
The final model is deployed using **Streamlit Cloud**.  
👉 [https://iavinashpathak-customer-churn-prediction-app-streamlit-bu0v0t.streamlit.app/]

<img width="1901" height="726" alt="Screenshot 2025-11-07 115547" src="https://github.com/user-attachments/assets/73076fc0-23ce-4ead-ac38-dd2edd04c3f8" />


---

## ⚙️ Project Workflow

The project follows the **standard machine learning lifecycle**:

### 1️⃣ Data Cleaning & Exploratory Data Analysis (EDA)
- Dataset used: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Fixed incorrect data types (e.g., `TotalCharges` was stored as object instead of float).
- Standardized inconsistent categorical values (e.g., replaced `"No internet service"` with `"No"`).
- Explored relationships between features and target variable: **Churn**.

### 2️⃣ Preprocessing & Feature Engineering
Built a complete **Scikit-learn pipeline** using `ColumnTransformer`:

| Feature Type | Transformations Applied |
|--------------|--------------------------|
| Numeric (`tenure`, `MonthlyCharges`, `TotalCharges`) | Missing value imputation + StandardScaler |
| Categorical (all others) | OneHotEncoding |

➡️ Result: A single automated preprocessing flow for both training & inference ✅

### 3️⃣ Model Training & Tuning
Models tested:
| Model | Notes |
|--------|-------|
| Logistic Regression | Baseline model |
| Decision Tree / Random Forest | High overfitting (99% train vs ~75% test accuracy) |
| ✅ **XGBoost (Final)** | Best performance & generalization |

🔍 **Hyperparameter Tuning**  
Performed using `GridSearchCV` on:
- `max_depth`
- `learning_rate`
- `n_estimators`

📌 **Final Model:**  
Trained and saved using Joblib as:  
`churn_predictor_pipeline.joblib`

### 4️⃣ Deployment
| Component | Description |
|-----------|-------------|
| Backend | `predictor.py` loads the saved model & exposes prediction function |
| Frontend | `app_streamlit.py` – Streamlit UI to take user input |
| Hosting | Streamlit Cloud connected directly to GitHub repo |

---

## 🛠️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/iavinashpathak/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app_streamlit.py
