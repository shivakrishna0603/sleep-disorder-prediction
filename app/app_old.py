# ============================================================
# app.py — Streamlit Web App
# Sleep Disorder Prediction Using Wearable Sensor Data
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="🛌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        color: #1a237e; text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.4rem; font-weight: bold;
    }
    .none      { background: #e8f5e9; color: #2e7d32; border: 2px solid #4caf50; }
    .insomnia  { background: #fff3e0; color: #e65100; border: 2px solid #ff9800; }
    .apnea     { background: #fce4ec; color: #880e4f; border: 2px solid #e91e63; }
</style>
""", unsafe_allow_html=True)

# ── Load Model Artifacts ──────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def load_artifacts():
    model     = pickle.load(open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb'))
    scaler    = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))
    le_target = pickle.load(open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb'))
    features  = pickle.load(open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb'))
    return model, scaler, le_target, features

try:
    model, scaler, le_target, feature_names = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model not found. Please run `02_model_training.py` first.\n\nError: {e}")

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-header">🛌 Sleep Disorder Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Using Wearable Sensor Data & Machine Learning</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar: About ────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sleeping.png", width=80)
    st.title("About This App")
    st.markdown("""
    This app uses **Machine Learning** to predict sleep disorders based on health and lifestyle data.

    **Disorders Detected:**
    - 🟢 None
    - 🟠 Insomnia
    - 🔴 Sleep Apnea

    **How to use:**
    1. Fill in your health data on the right
    2. Click **Predict**
    3. See your result + confidence

    ---
    **Dataset:** Sleep Health & Lifestyle Dataset (Kaggle)

    **Models Tried:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost
    """)

# ── Input Form ────────────────────────────────────────────────
st.subheader("📋 Enter Your Health Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal Info**")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age    = st.slider("Age", 18, 80, 30)
    occupation = st.selectbox("Occupation", [
        "Software Engineer", "Doctor", "Sales Representative",
        "Teacher", "Nurse", "Engineer", "Accountant",
        "Scientist", "Lawyer", "Manager", "Other"
    ])

with col2:
    st.markdown("**❤️ Health Metrics**")
    sleep_duration   = st.slider("Sleep Duration (hrs)", 4.0, 10.0, 7.0, 0.1)
    quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
    heart_rate       = st.slider("Heart Rate (bpm)", 50, 100, 72)
    systolic_bp      = st.slider("Systolic BP (mmHg)", 90, 180, 120)
    diastolic_bp     = st.slider("Diastolic BP (mmHg)", 60, 120, 80)

with col3:
    st.markdown("**🏃 Lifestyle Metrics**")
    physical_activity = st.slider("Physical Activity Level (min/day)", 0, 120, 45)
    stress_level      = st.slider("Stress Level (1-10)", 1, 10, 5)
    bmi_category      = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Normal Weight"])
    daily_steps       = st.slider("Daily Steps", 1000, 20000, 8000, 500)

# ── Prediction ────────────────────────────────────────────────
st.divider()

if st.button("🔍 Predict Sleep Disorder", type="primary", use_container_width=True):
    if not model_loaded:
        st.error("Model not loaded. Please train the model first.")
    else:
        # Encode inputs the same way as training
        gender_enc   = 0 if gender == "Male" else 1
        bmi_map      = {"Normal": 0, "Normal Weight": 1, "Obese": 2, "Overweight": 3}
        bmi_enc      = bmi_map.get(bmi_category, 0)

        occ_map = {
            "Accountant": 0, "Doctor": 1, "Engineer": 2, "Lawyer": 3,
            "Manager": 4, "Nurse": 5, "Other": 6, "Sales Representative": 7,
            "Scientist": 8, "Software Engineer": 9, "Teacher": 10
        }
        occ_enc = occ_map.get(occupation, 9)

        # Build input row — must match feature_names order from training
        input_dict = {
            'Gender': gender_enc,
            'Age': age,
            'Occupation': occ_enc,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'Physical Activity Level': physical_activity,
            'Stress Level': stress_level,
            'BMI Category': bmi_enc,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Systolic_BP': systolic_bp,
            'Diastolic_BP': diastolic_bp,
        }

        # Build dataframe with correct column order
        input_df = pd.DataFrame([input_dict])
        # Reindex to match training feature order
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        # Scale & predict
        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)[0]
        proba        = model.predict_proba(input_scaled)[0]
        label        = le_target.inverse_transform([prediction])[0]

        # ── Result Display ─────────────────────────────────────
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            css_cls = {"None": "none", "Insomnia": "insomnia", "Sleep Apnea": "apnea"}.get(label, "none")
            emoji   = {"None": "🟢", "Insomnia": "🟠", "Sleep Apnea": "🔴"}.get(label, "🟢")
            st.markdown(f"""
            <div class="result-box {css_cls}">
                {emoji} Predicted Disorder: <b>{label}</b><br>
                <span style="font-size:0.9rem; font-weight:normal">
                Confidence: {max(proba)*100:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Advice
            advice = {
                "None":        "✅ Your sleep patterns look healthy! Maintain your routine.",
                "Insomnia":    "⚠️ Signs of insomnia detected. Consider a consistent sleep schedule, reduce screen time before bed, and consult a doctor if symptoms persist.",
                "Sleep Apnea": "🚨 Possible sleep apnea detected. Please consult a healthcare professional for a proper sleep study."
            }
            st.info(advice.get(label, ""))

        with res_col2:
            # Probability bar chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            classes = le_target.classes_
            colors  = ['#4CAF50' if c == 'None' else '#FF9800' if c == 'Insomnia' else '#E91E63' for c in classes]
            ax.barh(classes, proba * 100, color=colors, edgecolor='black')
            ax.set_xlabel('Probability (%)')
            ax.set_title('Prediction Confidence')
            ax.set_xlim(0, 100)
            for i, (v, c) in enumerate(zip(proba * 100, classes)):
                ax.text(v + 1, i, f'{v:.1f}%', va='center')
            plt.tight_layout()
            st.pyplot(fig)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style="text-align:center; color:#888; font-size:0.85rem;">
⚠️ This tool is for educational purposes only. Always consult a healthcare professional for medical advice.<br>
B.Tech Final Year Mini Project | Sleep Disorder Prediction Using Wearable Sensor Data
</p>
""", unsafe_allow_html=True)
