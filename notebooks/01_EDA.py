# ============================================================
# 01_EDA.py — Exploratory Data Analysis
# Sleep Disorder Prediction Using Wearable Sensor Data
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Load Dataset ─────────────────────────────────────────────
# Dataset: Sleep Health and Lifestyle Dataset (Kaggle)
# Download from: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
# Save as: data/sleep_health.csv

df = pd.read_csv('../data/sleep_health.csv')

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape        : {df.shape}")
print(f"Columns      : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nTarget Distribution:\n{df['Sleep Disorder'].value_counts()}")

# ── Basic Stats ───────────────────────────────────────────────
print("\nDescriptive Statistics:")
print(df.describe())

# ── Visualizations ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Sleep Disorder EDA', fontsize=16, fontweight='bold')

# 1. Target Distribution
ax = axes[0, 0]
df['Sleep Disorder'].value_counts().plot(kind='bar', ax=ax, color=['#4CAF50','#FF5722','#2196F3'], edgecolor='black')
ax.set_title('Sleep Disorder Distribution')
ax.set_xlabel('Disorder Type')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)

# 2. Sleep Duration Distribution
ax = axes[0, 1]
df['Sleep Duration'].hist(ax=ax, bins=20, color='#2196F3', edgecolor='black')
ax.set_title('Sleep Duration Distribution')
ax.set_xlabel('Hours')
ax.set_ylabel('Frequency')

# 3. Heart Rate by Disorder
ax = axes[0, 2]
df.boxplot(column='Heart Rate', by='Sleep Disorder', ax=ax)
ax.set_title('Heart Rate by Sleep Disorder')
ax.set_xlabel('Sleep Disorder')
plt.sca(ax)
plt.xticks(rotation=0)

# 4. Age Distribution
ax = axes[1, 0]
df['Age'].hist(ax=ax, bins=20, color='#FF9800', edgecolor='black')
ax.set_title('Age Distribution')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')

# 5. Stress Level vs Sleep Quality
ax = axes[1, 1]
sns.scatterplot(data=df, x='Stress Level', y='Quality of Sleep',
                hue='Sleep Disorder', ax=ax, palette='Set2')
ax.set_title('Stress Level vs Sleep Quality')

# 6. Physical Activity vs Sleep Duration
ax = axes[1, 2]
sns.scatterplot(data=df, x='Physical Activity Level', y='Sleep Duration',
                hue='Sleep Disorder', ax=ax, palette='Set1')
ax.set_title('Physical Activity vs Sleep Duration')

plt.tight_layout()
plt.savefig('../report/eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[✓] EDA plots saved to report/eda_plots.png")

# ── Correlation Heatmap ───────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../report/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("[✓] Correlation heatmap saved to report/correlation_heatmap.png")
