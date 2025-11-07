import streamlit as st
import predictor  
import pandas as pd

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="👋",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")
st.markdown("Enter the customer's details below. The model will predict the probability of churn.")


with st.form(key="churn_form"):
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Info")
        gender = st.selectbox(
            "Gender", 
            ("Male", "Female"), 
            key="gender"
        )
        SeniorCitizen = st.selectbox(
            "Senior Citizen", 
            ("No", "Yes"), 
            key="SeniorCitizen"
        )
        Partner = st.selectbox(
            "Has Partner", 
            ("No", "Yes"), 
            key="Partner"
        )
        Dependents = st.selectbox(
            "Has Dependents", 
            ("No", "Yes"), 
            key="Dependents"
        )

    # --- Column 2: Account Info ---
    with col2:
        st.subheader("Account Info")
        tenure = st.number_input(
            "Tenure (months)", 
            min_value=0, 
            max_value=100, 
            value=1, 
            step=1, 
            key="tenure"
        )
        Contract = st.selectbox(
            "Contract Type", 
            ("Month-to-month", "One year", "Two year"), 
            key="Contract"
        )
        PaperlessBilling = st.selectbox(
            "Paperless Billing", 
            ("No", "Yes"), 
            key="PaperlessBilling"
        )
        PaymentMethod = st.selectbox(
            "Payment Method", 
            ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"), 
            key="PaymentMethod"
        )

    # --- Column 3: Service Charges ---
    with col3:
        st.subheader("Charges")
        MonthlyCharges = st.number_input(
            "Monthly Charges ($)", 
            min_value=0.0, 
            value=50.0, 
            step=0.05, 
            key="MonthlyCharges"
        )
        TotalCharges = st.number_input(
            "Total Charges ($)", 
            min_value=0.0, 
            value=100.0, 
            step=0.1, 
            key="TotalCharges"
        )

    # --- Service Details (Full Width) ---
    st.subheader("Subscribed Services")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        PhoneService = st.selectbox("Phone Service", ("No", "Yes"), key="PhoneService")
        MultipleLines = st.selectbox("Multiple Lines", ("No", "Yes", "No phone service"), key="MultipleLines")
        InternetService = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"), key="InternetService")

    with col_s2:
        OnlineSecurity = st.selectbox("Online Security", ("No", "Yes", "No internet service"), key="OnlineSecurity")
        OnlineBackup = st.selectbox("Online Backup", ("No", "Yes", "No internet service"), key="OnlineBackup")
        DeviceProtection = st.selectbox("Device Protection", ("No", "Yes", "No internet service"), key="DeviceProtection")

    with col_s3:
        TechSupport = st.selectbox("Tech Support", ("No", "Yes", "No internet service"), key="TechSupport")
        StreamingTV = st.selectbox("Streaming TV", ("No", "Yes", "No internet service"), key="StreamingTV")
        StreamingMovies = st.selectbox("Streaming Movies", ("No", "Yes", "No internet service"), key="StreamingMovies")

    # --- Submit Button ---
    st.divider()
    submitted = st.form_submit_button("Predict Churn")

# --- After Form Submission ---
if submitted:
    # 1. Collect all inputs into a dictionary (matching the training columns)
    raw_data_dict = {
        'gender': gender,
        'SeniorCitizen': 1 if SeniorCitizen == 'Yes' else 0, # Convert 'Yes'/'No' to 1/0
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        if raw_data_dict[col] == 'No internet service':
            raw_data_dict[col] = 'No'
    
    # Same for 'No phone service'
    if raw_data_dict['MultipleLines'] == 'No phone service':
        raw_data_dict['MultipleLines'] = 'No'

    # 3. Call the predictor function
    with st.spinner("Analyzing customer data..."):
        result = predictor.predict_churn(raw_data_dict)

    # 4. Display the results
    st.subheader("Prediction Result")
    if result['status'] == 'OK':
        prob_yes = result['probability_yes']
        
        if result['churn'] == 'Yes':
            st.error(f"**Prediction: Customer will CHURN**")
        else:
            st.success(f"**Prediction: Customer will STAY**")

        # Display the probability with a metric and a progress bar
        st.metric(
            label="Probability of Churn",
            value=f"{prob_yes:.2%}"
        )
        
        st.progress(prob_yes, text=f"{prob_yes:.2%} Risk")
        
        st.expander("Show Raw Prediction Data").json(result)

    else:
        st.error(f"Error in prediction: {result['error']}")