import streamlit as st
import pandas as pd
import pickle
import joblib

# 1. Load the "Brain" we saved earlier
model = joblib.load('customer_ai.pkl')
# 2. Setup the Website Title
st.title("📊 Telecom Customer Churn Predictor")
st.write("Enter customer details below to see if they are likely to leave.")

# 3. Create Input Fields for the User
st.header("Customer Information")

# We use the three features our AI learned: tenure, MonthlyCharges, and TotalCharges
tenure = st.slider("Tenure (Months with company)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

# 4. Make the Prediction
# We put the inputs into a small table (DataFrame) for the AI to read
input_data = pd.DataFrame([[tenure, monthly_charges, total_charges]], 
                          columns=['tenure', 'MonthlyCharges', 'TotalCharges'])

if st.button("Predict Risk"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] # Get the percentage risk

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk! Probability of leaving: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk. Probability of leaving: {probability:.2%}")


st.info("This AI uses the XGBoost model saved in your customer_ai.pkl file.")
