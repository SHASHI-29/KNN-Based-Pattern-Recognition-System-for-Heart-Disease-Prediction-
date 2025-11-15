# ------------------------------------------------------------
# Streamlit App: KNN-Based Heart Disease Prediction System
# ------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💓 Heart Disease Prediction System (KNN)")
st.write("Enter the medical details below to predict heart disease risk.")

# ------------------------------------------------------------
# Input Fields (13 features of Heart Disease Dataset)
# ------------------------------------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["Female (0)", "Male (1)"])
sex = 1 if sex == "Male (1)" else 0

cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=230)
fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", ["No (0)", "Yes (1)"])
fbs = 1 if fbs == "Yes (1)" else 0

restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina", ["No (0)", "Yes (1)"])
exang = 1 if exang == "Yes (1)" else 0

oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [1, 2, 3])

# ------------------------------------------------------------
# Convert Input to numpy array
# ------------------------------------------------------------
user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

scaled_input = scaler.transform(user_input)

# ------------------------------------------------------------
# Prediction Button
# ------------------------------------------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(scaled_input)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ **High Risk of Heart Disease**")
    else:
        st.success("✅ **Low Risk of Heart Disease**")