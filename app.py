import streamlit as st
import pandas as pd
import joblib

model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('feature_columns.pkl')

st.title("Heart Disease Prediction by Mateen")

st.markdown("Provide the following details to predict the likelihood of heart disease:")
age = st.slider("Age", 18,100,40)
sex = st.selectbox("SEX", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ['N', 'Y'])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex' + sex  : 1,
        'ChestPainType' + chest_pain :1,
        'RestingECG' + resting_ecg :1,
        'ExerciseAngina' + exercise_angina :1,
        'ST_Slope' + st_slope :1
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]    

    scaled_input = scaler.transform(input_df) 
    prediction = model.predict(scaled_input)[0]   

    if prediction == 1:
        st.error("The model predicts a high likelihood of heart disease. Please consult a healthcare professional.")
    else:
        st.success("The model predicts a low likelihood of heart disease. However, always consult a healthcare professional for accurate diagnosis and advice.")    