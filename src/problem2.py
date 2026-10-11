import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Directories
DATA_PATH = "data/insurance.csv"
FIGURES_DIR = "results/figures"
TABLES_DIR = "results/tables"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("PROBLEM 2: LINEAR & LOGISTIC REGRESSION ANALYSIS")
print("=" * 65)

# -------------------------------------------------------------------------
# PART (a): Two-Panel Scatter Plot with Linear Regression Fits
# -------------------------------------------------------------------------
print("\n--- Part (a): Linear Regression (Age & BMI vs Hospital Charges) ---")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Define smoker colors
hue_colors = {'yes': 'crimson', 'no': 'royalblue'}

# 1. Subplot 1: Age vs Charges
sns.scatterplot(
    data=df, x='age', y='charges', hue='smoker', palette=hue_colors,
    alpha=0.6, ax=axes[0]
)

# Linear regression for Age vs Charges
X_age = sm.add_constant(df['age'])
model_age = sm.OLS(df['charges'], X_age).fit()
intercept_age, slope_age = model_age.params['const'], model_age.params['age']

# Add regression line
age_grid = np.linspace(df['age'].min(), df['age'].max(), 100)
axes[0].plot(age_grid, intercept_age + slope_age * age_grid, color='black', linestyle='--', linewidth=2,
             label=f'Fit: Y = {slope_age:.2f}X + {intercept_age:.2f}\n(R² = {model_age.rsquared:.3f})')

axes[0].set_title("Age vs. Hospital Charges", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Age (years)", fontsize=11)
axes[0].set_ylabel("Hospital Charges ($)", fontsize=11)
axes[0].legend(title="Smoker", frameon=True)

print(f"Age vs Charges Fit Params: Slope = {slope_age:.4f}, Intercept = {intercept_age:.4f}, R² = {model_age.rsquared:.4f}")

# 2. Subplot 2: BMI vs Charges
sns.scatterplot(
    data=df, x='bmi', y='charges', hue='smoker', palette=hue_colors,
    alpha=0.6, ax=axes[1]
)

# Linear regression for BMI vs Charges
X_bmi = sm.add_constant(df['bmi'])
model_bmi = sm.OLS(df['charges'], X_bmi).fit()
intercept_bmi, slope_bmi = model_bmi.params['const'], model_bmi.params['bmi']

# Add regression line
bmi_grid = np.linspace(df['bmi'].min(), df['bmi'].max(), 100)
axes[1].plot(bmi_grid, intercept_bmi + slope_bmi * bmi_grid, color='black', linestyle='--', linewidth=2,
             label=f'Fit: Y = {slope_bmi:.2f}X + {intercept_bmi:.2f}\n(R² = {model_bmi.rsquared:.3f})')

axes[1].set_title("BMI vs. Hospital Charges", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Body Mass Index (BMI)", fontsize=11)
axes[1].set_ylabel("Hospital Charges ($)", fontsize=11)
axes[1].legend(title="Smoker", frameon=True)

print(f"BMI vs Charges Fit Params: Slope = {slope_bmi:.4f}, Intercept = {intercept_bmi:.4f}, R² = {model_bmi.rsquared:.4f}")

plt.suptitle("Problem 2(a): Patient Age and BMI vs Hospital Charges by Smoker Status", fontsize=14, fontweight='bold')
plt.tight_layout()

fig_path = f"{FIGURES_DIR}/problem2_scatter_regressions.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Two-panel scatter plot saved to: {fig_path}")

# -------------------------------------------------------------------------
# PART (b): Logistic Regression Predicting Smoker Status
# -------------------------------------------------------------------------
print("\n--- Part (b): Logistic Regression (Predicting Smoker Status) ---")

# Feature matrix X and target y
X = df[['age', 'bmi', 'charges']]
y = (df['smoker'] == 'yes').astype(int)  # 1 = smoker, 0 = non-smoker

# -------------------------------------------------------------------------
# Case 1: Training and Testing on the Same Data (100% / 100%)
# -------------------------------------------------------------------------
print("\n[Case 1]: Training and Testing on FULL Dataset (100% / 100%)")
clf_same = LogisticRegression(random_state=10, max_iter=1000)
clf_same.fit(X, y)
y_pred_same = clf_same.predict(X)

cm_same = confusion_matrix(y, y_pred_same)
cr_same = classification_report(y, y_pred_same, target_names=['Non-Smoker', 'Smoker'], digits=4)
acc_same = clf_same.score(X, y)

print("Confusion Matrix:")
print(pd.DataFrame(cm_same, index=['True Non-Smoker', 'True Smoker'], columns=['Pred Non-Smoker', 'Pred Smoker']))
print("\nClassification Report:")
print(cr_same)
print(f"Overall Accuracy (Same Data): {acc_same * 100:.2f}%")

# -------------------------------------------------------------------------
# Case 2: 75% Train / 25% Test Split (random_state=10)
# -------------------------------------------------------------------------
print("\n[Case 2]: Train on 75% / Test on 25% Split (random_state=10)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10, stratify=y)

clf_split = LogisticRegression(random_state=10, max_iter=1000)
clf_split.fit(X_train, y_train)
y_pred_split = clf_split.predict(X_test)

cm_split = confusion_matrix(y_test, y_pred_split)
cr_split = classification_report(y_test, y_pred_split, target_names=['Non-Smoker', 'Smoker'], digits=4)
acc_split = clf_split.score(X_test, y_test)

print("Confusion Matrix (Test Set):")
print(pd.DataFrame(cm_split, index=['True Non-Smoker', 'True Smoker'], columns=['Pred Non-Smoker', 'Pred Smoker']))
print("\nClassification Report (Test Set):")
print(cr_split)
print(f"Overall Accuracy (75/25 Split): {acc_split * 100:.2f}%")

print("=" * 65)