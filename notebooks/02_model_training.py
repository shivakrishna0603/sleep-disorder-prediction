# ============================================================
# 02_model_training.py — Preprocessing + Model Training (v2, tuned)
# Sleep Disorder Prediction Using Wearable Sensor Data
# ============================================================

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv('../data/sleep_health.csv')
print(f"[OK] Loaded dataset: {df.shape}")

# ── 2. Preprocessing ──────────────────────────────────────────
if 'Person ID' in df.columns:
    df.drop('Person ID', axis=1, inplace=True)
if 'Blood Pressure' in df.columns:
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    df.drop('Blood Pressure', axis=1, inplace=True)
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

le_dict = {}
cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'Sleep Disorder']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

le_target = LabelEncoder()
df['Sleep Disorder'] = le_target.fit_transform(df['Sleep Disorder'])
print(f"[OK] Target classes: {list(le_target.classes_)}")

# ── 3. Feature / Target Split ─────────────────────────────────
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']
feature_names = X.columns.tolist()
print(f"[OK] Features ({len(feature_names)}): {feature_names}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[OK] Train: {X_train.shape}, Test: {X_test.shape}")

# ── 4. Hyperparameter-tuned model comparison ────────────────────
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grids = {
    'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=42),
                             {'C': [0.1, 1, 10], 'solver': ['lbfgs']}),
    'Random Forest': (RandomForestClassifier(random_state=42),
                       {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42),
                          {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [2, 3]}),
    'SVM': (SVC(probability=True, random_state=42),
            {'C': [1, 10], 'kernel': ['rbf', 'linear']}),
}
if XGBOOST_AVAILABLE:
    param_grids['XGBoost'] = (XGBClassifier(eval_metric='mlogloss', random_state=42),
                               {'n_estimators': [100, 200], 'max_depth': [3, 4, 5], 'learning_rate': [0.05, 0.1]})

results = {}
print("\n" + "=" * 50)
print("MODEL COMPARISON (grid-searched, 5-fold CV)")
print("=" * 50)

for name, (estimator, grid) in param_grids.items():
    gs = GridSearchCV(estimator, grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    best_est = gs.best_estimator_
    y_pred = best_est.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'model': best_est, 'accuracy': acc, 'cv_accuracy': gs.best_score_,
                      'y_pred': y_pred, 'params': gs.best_params_}
    print(f"\n{name}  (best params: {gs.best_params_})")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  CV  Accuracy  : {gs.best_score_:.4f}")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# ── 5. Best Model — ranked by test accuracy, tie-broken by CV accuracy ──
best_name = max(results, key=lambda k: (results[k]['accuracy'], results[k]['cv_accuracy']))
best_model = results[best_name]['model']
print(f"\n[BEST] {best_name}  (Test Accuracy: {results[best_name]['accuracy']:.4f}, CV Accuracy: {results[best_name]['cv_accuracy']:.4f})")

# ── 6. Confusion Matrix + Model Comparison plot ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Best Model: {best_name}', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, results[best_name]['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_target.classes_)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix')

acc_vals = [results[n]['accuracy'] for n in results]
ax = axes[1]
bars = ax.barh(list(results.keys()), acc_vals, color='#2196F3', edgecolor='black')
ax.set_xlim(0, 1)
ax.set_xlabel('Test Accuracy')
ax.set_title('Model Accuracy Comparison (tuned)')
for bar, val in zip(bars, acc_vals):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('../report/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Model comparison plot saved")

# ── 7. Feature Importance ─────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    fi.plot(kind='bar', color='#FF9800', edgecolor='black')
    plt.title(f'Feature Importance — {best_name}', fontsize=13, fontweight='bold')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('../report/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Feature importance plot saved")

# ── 8. Save Artifacts ─────────────────────────────────────────
pickle.dump(best_model, open('../models/best_model.pkl', 'wb'))
pickle.dump(scaler,     open('../models/scaler.pkl', 'wb'))
pickle.dump(le_target,  open('../models/label_encoder.pkl', 'wb'))
pickle.dump(le_dict,    open('../models/cat_encoders.pkl', 'wb'))
pickle.dump(feature_names, open('../models/feature_names.pkl', 'wb'))

meta = {'model_name': best_name, 'test_accuracy': results[best_name]['accuracy'],
        'cv_accuracy': results[best_name]['cv_accuracy'], 'params': results[best_name]['params']}
pickle.dump(meta, open('../models/model_meta.pkl', 'wb'))

print(f"\n[OK] All artifacts saved to models/  (best: {best_name}, test acc {results[best_name]['accuracy']:.4f})")
