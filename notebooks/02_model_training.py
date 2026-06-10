# ============================================================
# 02_model_training.py — Preprocessing + Model Training
# Sleep Disorder Prediction Using Wearable Sensor Data
# ============================================================

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[!] XGBoost not installed. Skipping XGBoost model.")

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv('../data/sleep_health.csv')
print(f"[✓] Loaded dataset: {df.shape}")

# ── 2. Preprocessing ──────────────────────────────────────────

# Drop Person ID if present
if 'Person ID' in df.columns:
    df.drop('Person ID', axis=1, inplace=True)

# Split 'Blood Pressure' into Systolic and Diastolic
if 'Blood Pressure' in df.columns:
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    df.drop('Blood Pressure', axis=1, inplace=True)

# Fill missing Sleep Disorder with 'None'
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# Encode categorical columns
le_dict = {}
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != 'Sleep Disorder']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Encode target
le_target = LabelEncoder()
df['Sleep Disorder'] = le_target.fit_transform(df['Sleep Disorder'])
print(f"[✓] Target classes: {list(le_target.classes_)}")

# ── 3. Feature / Target Split ─────────────────────────────────
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']

feature_names = X.columns.tolist()
print(f"[✓] Features ({len(feature_names)}): {feature_names}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[✓] Train: {X_train.shape}, Test: {X_test.shape}")

# ── 4. Train Models ───────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}
if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(use_label_encoder=False,
                                      eval_metric='mlogloss', random_state=42)

results = {}
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy').mean()
    results[name] = {'model': model, 'accuracy': acc, 'cv_accuracy': cv, 'y_pred': y_pred}
    print(f"\n{name}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  CV  Accuracy  : {cv:.4f}")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# ── 5. Best Model ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
print(f"\n[★] Best Model: {best_name}  (Accuracy: {results[best_name]['accuracy']:.4f})")

# ── 6. Confusion Matrix ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Best Model: {best_name}', fontsize=14, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_target.classes_)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix')

# Model accuracy comparison
acc_vals = [results[n]['accuracy'] for n in results]
ax = axes[1]
bars = ax.barh(list(results.keys()), acc_vals, color='#2196F3', edgecolor='black')
ax.set_xlim(0, 1)
ax.set_xlabel('Test Accuracy')
ax.set_title('Model Accuracy Comparison')
for bar, val in zip(bars, acc_vals):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('../report/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("[✓] Model comparison plot saved")

# ── 7. Feature Importance ─────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    fi.plot(kind='bar', color='#FF9800', edgecolor='black')
    plt.title(f'Feature Importance — {best_name}', fontsize=13, fontweight='bold')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('../report/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[✓] Feature importance plot saved")

# ── 8. Save Artifacts ─────────────────────────────────────────
pickle.dump(best_model, open('../models/best_model.pkl', 'wb'))
pickle.dump(scaler,     open('../models/scaler.pkl', 'wb'))
pickle.dump(le_target,  open('../models/label_encoder.pkl', 'wb'))
pickle.dump(le_dict,    open('../models/cat_encoders.pkl', 'wb'))
pickle.dump(feature_names, open('../models/feature_names.pkl', 'wb'))

print("\n[✓] All artifacts saved to models/")
print(f"    best_model.pkl   → {best_name}")
print(f"    scaler.pkl       → StandardScaler")
print(f"    label_encoder.pkl→ Target classes: {list(le_target.classes_)}")
