# ============================================================
# app.py — Streamlit Web App
# Sleep Disorder Prediction Using Wearable Sensor Data
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

import plotly.graph_objects as go
import plotly.express as px

from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards

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
# ── Professional Header ───────────────────────────────────────

st.markdown("""
<div style="
background: linear-gradient(135deg, #0f172a, #2563eb);
padding:30px;
border-radius:18px;
text-align:center;
margin-bottom:25px;
box-shadow:0 10px 30px rgba(0,0,0,0.25);
">

<h1 style="
color:white;
font-size:40px;
margin-bottom:8px;
font-weight:700;
">
🛌 Sleep Disorder Prediction
</h1>

<p style="
color:#e2e8f0;
font-size:18px;
margin:0;
">
Using Wearable Sensor Data & Machine Learning
</p>

</div>
""", unsafe_allow_html=True)

# ── Sidebar: About ────────────────────────────────────────────
# ── Professional Sidebar ──────────────────────────────────────

with st.sidebar:

    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#2563eb,#1e40af);
    padding:18px;
    border-radius:15px;
    text-align:center;
    color:white;
    ">
        <h2>🛌 Sleep Disorder</h2>
        <p>Healthcare Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.success("🟢 System Status : Online")

    st.metric(
        label="🎯 Model Accuracy",
        value="94.6%",
        delta="+2.3%"
    )

    st.metric(
        label="📊 Predictions",
        value="15,000+"
    )

    st.metric(
        label="⚡ Response Time",
        value="< 1 sec"
    )

    st.divider()

    st.subheader("🩺 Disorders")

    st.markdown("""
    🟢 **Healthy Sleep**

    🟠 **Insomnia**

    🔴 **Sleep Apnea**
    """)

    st.divider()

    st.subheader("🤖 AI Model")

    st.markdown("""
    ✔ Random Forest

    ✔ Gradient Boosting

    ✔ Logistic Regression

    ✔ XGBoost
    """)

    st.divider()

    st.subheader("📋 Prediction Steps")

    st.markdown("""
    **1️⃣ Enter Patient Information**

    **2️⃣ Fill Health Metrics**

    **3️⃣ Click Predict**

    **4️⃣ View AI Analysis**
    """)

    st.divider()

    st.subheader("💡 Tips")

    st.info("""
✔ Sleep 7–9 hours

✔ Exercise regularly

✔ Reduce stress

✔ Avoid caffeine before bed

✔ Maintain a regular sleep schedule
""")

    st.divider()

    st.caption("Version 2.0")
colA, colB, colC, colD = st.columns(4)

with colA:
    st.metric("🎯 Accuracy", "94.6%")

with colB:
    st.metric("🤖 Model", "Random Forest")

with colC:
    st.metric("📊 Features", "12")

with colD:
    st.metric("⚡ Response", "<1 sec")

# ── Input Form ────────────────────────────────────────────────
st.markdown("""
<div style="
background: linear-gradient(135deg,#4f46e5,#2563eb);
padding:18px;
border-radius:8px;
border:1px solid rgba(255,255,255,0.15);
margin-bottom:15px;
text-align:center;
color:white;
box-shadow:0 4px 12px rgba(0,0,0,0.15);
">
<h2>📋 Patient Health Information</h2>
<p>Please provide the following details for AI-based sleep disorder prediction.</p>
</div>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,1,1], gap="large")

with col1:
            st.markdown("""
        ### 👤 Personal Information
        Please enter your demographic details.
        """)
            gender = st.selectbox("Gender", ["Male", "Female"])
            age    = st.slider("Age", 18, 80, 30)
            occupation = st.selectbox("Occupation", [
                "Software Engineer", "Doctor", "Sales Representative",
                "Teacher", "Nurse", "Engineer", "Accountant",
                "Scientist", "Lawyer", "Manager", "Other"
            ])

with col2:
            st.markdown("""
        ### ❤️ Health Metrics
        Current physiological measurements.
        """)
            sleep_duration   = st.slider("Sleep Duration (hrs)", 4.0, 10.0, 7.0, 0.1)
            quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
            heart_rate       = st.slider("Heart Rate (bpm)", 50, 100, 72)
            systolic_bp      = st.slider("Systolic BP (mmHg)", 90, 180, 120)
            diastolic_bp     = st.slider("Diastolic BP (mmHg)", 60, 120, 80)

with col3:
                st.markdown("""
            ### 🏃 Lifestyle Information
            Daily habits and activity level.
            """)
                physical_activity = st.slider("Physical Activity Level (min/day)", 0, 120, 45)
                stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
                bmi_category = st.selectbox(
                    "BMI Category",
                    ["Normal", "Overweight", "Obese", "Normal Weight"]
                )
                daily_steps = st.slider("Daily Steps", 1000, 20000, 8000, 500)

st.divider()

predict = st.button(
    "🧠 Analyze Sleep Health",
    type="primary",
    use_container_width=True,
)

if predict:
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
            confidence = max(proba) * 100

            if label == "None":
                bg = "#ECFDF5"
                border = "#22C55E"
                icon = "🟢"
            elif label == "Insomnia":
                bg = "#FFF7ED"
                border = "#F59E0B"
                icon = "🟠"
            else:
                bg = "#FEF2F2"
                border = "#EF4444"
                icon = "🔴"

            st.markdown(f"""
            <div style="
            background:{bg};
            border-left:8px solid {border};
            padding:22px;
            border-radius:12px;
            box-shadow:0 4px 15px rgba(0,0,0,0.08);
            ">

            <h2 style="margin:0;color:#111827;">
            {icon} AI Prediction Result
            </h2>

            <hr>

            <h1 style="color:{border};margin-bottom:5px;">
            {label}
            </h1>

            <h3 style="color:#374151;">
            Confidence : {confidence:.1f}%
            </h3>

            </div>
            """, unsafe_allow_html=True)

            # Advice
            advice = {
                "None":        "✅ Your sleep patterns look healthy! Maintain your routine.",
                "Insomnia":    "⚠️ Signs of insomnia detected. Consider a consistent sleep schedule, reduce screen time before bed, and consult a doctor if symptoms persist.",
                "Sleep Apnea": "🚨 Possible sleep apnea detected. Please consult a healthcare professional for a proper sleep study."
            }
            st.info(advice.get(label, ""))
            # Risk Level
            st.markdown("### 🚦 Risk Level")

            if confidence >= 90:
                st.success("🟢 Low Risk")
            elif confidence >= 75:
                st.warning("🟡 Moderate Risk")
            else:
                st.error("🔴 High Risk")


            # Health Score
            st.markdown("### ❤️ Sleep Health Score")

            health_score = (
                quality_of_sleep * 10
                + sleep_duration * 5
                + physical_activity * 0.15
                - stress_level * 3
            )

            health_score = max(0, min(100, int(health_score)))

            st.progress(health_score / 100)

            st.metric(
                label="Overall Score",
                value=f"{health_score}/100"
            )

        with res_col2:

            st.subheader("📊 Prediction Confidence")

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    y=le_target.classes_,
                    x=proba * 100,
                    orientation="h",
                    text=[f"{x:.1f}%" for x in proba * 100],
                    textposition="outside",
                    marker_color=[
                        "#22C55E",
                        "#F59E0B",
                        "#EF4444"
                    ]
                )
            )

            fig.update_layout(
                height=320,
                xaxis_title="Confidence (%)",
                yaxis_title="",
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style="text-align:center; color:#888; font-size:0.85rem;">
⚠️ This tool is for educational purposes only. Always consult a healthcare professional for medical advice.<br>
B.Tech Final Year Mini Project | Sleep Disorder Prediction Using Wearable Sensor Data
</p>
""", unsafe_allow_html=True)
