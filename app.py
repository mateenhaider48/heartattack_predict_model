import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Custom CSS for a modern look
# ------------------------------------------------------------------
st.markdown("""
    <style>
        .main {
            padding-top: 1.5rem;
        }
        .stApp {
            background: linear-gradient(180deg, #fafafa 0%, #f4f6f8 100%);
        }
        h1 {
            font-weight: 800 !important;
            background: linear-gradient(90deg, #ff4b4b, #ff8080);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding-bottom: 0.2rem;
        }
        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.8rem;
        }
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
            color: white;
            font-weight: 700;
            font-size: 1rem;
            padding: 0.7rem 0;
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 14px rgba(255, 75, 75, 0.35);
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(255, 75, 75, 0.45);
        }
        div[data-testid="stMetric"] {
            background-color: white;
            border-radius: 12px;
            padding: 0.8rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .result-card {
            padding: 1.4rem;
            border-radius: 14px;
            margin-top: 1.2rem;
            font-size: 1.05rem;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_columns = joblib.load('feature_columns.pkl')
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write(
        "This tool uses a machine learning model to estimate the "
        "likelihood of heart disease based on clinical indicators. "
        "It is intended for educational purposes only and is **not** "
        "a substitute for professional medical advice."
    )
    st.markdown("---")
    st.markdown("**Built by Mateen** 🩺")

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.markdown('<p class="subtitle">Enter patient details below to estimate the likelihood of heart disease.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Input Form
# ------------------------------------------------------------------
with st.form("prediction_form"):
    st.markdown("#### 🧍 Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ['M', 'F'])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    with col2:
        chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
        resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ['N', 'Y'])
        st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

    st.markdown("")
    submitted = st.form_submit_button("🔍 Predict Risk")

# ------------------------------------------------------------------
# Prediction Logic
# ------------------------------------------------------------------
if submitted:
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex' + sex: 1,
        'ChestPainType' + chest_pain: 1,
        'RestingECG' + resting_ecg: 1,
        'ExerciseAngina' + exercise_angina: 1,
        'ST_Slope' + st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    with st.spinner("Analyzing patient data..."):
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        try:
            proba = model.predict_proba(scaled_input)[0][1]
        except AttributeError:
            proba = None

    st.markdown("---")
    st.markdown("#### 📊 Result")

    if proba is not None:
        m1, m2 = st.columns(2)
        m1.metric("Predicted Risk", "High" if prediction == 1 else "Low")
        m2.metric("Estimated Probability", f"{proba*100:.1f}%")
    else:
        st.metric("Predicted Risk", "High" if prediction == 1 else "Low")

    if prediction == 1:
        st.markdown(
            '<div class="result-card" style="background-color:#fee2e2; color:#991b1b; border:1px solid #fca5a5;">'
            '⚠️ <strong>High likelihood of heart disease detected.</strong><br>'
            'Please consult a healthcare professional for a thorough evaluation.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-card" style="background-color:#dcfce7; color:#166534; border:1px solid #86efac;">'
            '✅ <strong>Low likelihood of heart disease detected.</strong><br>'
            'Still, regular checkups with a healthcare professional are recommended.'
            '</div>',
            unsafe_allow_html=True
        )

    with st.expander("📋 View submitted data"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))