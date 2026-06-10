# 🛌 Sleep Disorder Prediction Using Wearable Sensor Data
**B.Tech Final Year Mini Project | Computer Science & IT**

---

## 📌 Problem Statement
Sleep disorders like insomnia and sleep apnea affect millions of people. Early prediction using
wearable sensor data (heart rate, activity, sleep patterns) can help avoid expensive clinical tests.

---

## 🗂️ Project Structure
```
sleep-disorder-prediction/
├── data/
│   └── sleep_health.csv          ← Place your dataset here
├── notebooks/
│   ├── 01_EDA.py                 ← Exploratory Data Analysis
│   └── 02_model_training.py      ← Preprocessing + Model Training
├── models/
│   ├── best_model.pkl            ← Saved best ML model
│   ├── scaler.pkl                ← StandardScaler
│   ├── label_encoder.pkl         ← Target LabelEncoder
│   ├── cat_encoders.pkl          ← Categorical encoders
│   └── feature_names.pkl         ← Feature column order
├── app/
│   └── app.py                    ← Streamlit Web App
├── report/
│   ├── eda_plots.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
1. Go to: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
2. Download `Sleep_health_and_lifestyle_dataset.csv`
3. Rename it to `sleep_health.csv`
4. Place it in the `data/` folder

### Step 3: Run EDA
```bash
cd notebooks
python 01_EDA.py
```
This generates plots in the `report/` folder.

### Step 4: Train the Model
```bash
python 02_model_training.py
```
This trains and saves the best model to `models/`.

### Step 5: Launch the Web App
```bash
cd app
streamlit run app.py
```
Open your browser at: http://localhost:8501

---

## 🧰 Tech Stack
| Component     | Tool |
|---------------|------|
| Language      | Python 3.10+ |
| ML Models     | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Web UI        | Streamlit |
| Notebook      | Google Colab / Jupyter |

---

## 🤖 Models Trained
| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest       | Ensemble, handles noise well |
| Gradient Boosting   | High accuracy, robust |
| XGBoost             | Best performance typically |

---

## 📊 Features Used
- Age, Gender, Occupation
- Sleep Duration, Quality of Sleep
- Physical Activity Level
- Stress Level, BMI Category
- Heart Rate, Daily Steps
- Systolic BP, Diastolic BP

---

## 🎯 Output Classes
- **None** — No sleep disorder detected
- **Insomnia** — Difficulty falling/staying asleep
- **Sleep Apnea** — Breathing interruptions during sleep

---

## ⚠️ Disclaimer
This tool is for educational purposes only. Always consult a healthcare professional for medical diagnosis.
