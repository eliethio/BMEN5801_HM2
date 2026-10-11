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
# PART (a): Two-Panel Scatter Plot with Grouped Linear Regression Fits
# -------------------------------------------------------------------------
print("\n--- Part (a): Grouped Linear Regression (Smoker vs Non-Smoker) ---")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

hue_colors = {'yes': 'crimson', 'no': 'royalblue'}
smoker_df = df[df['smoker'] == 'yes']
non_smoker_df = df[df['smoker'] == 'no']

# Helper function to fit OLS and return fit line data
def fit_ols_group(data, x_col, y_col='charges'):
    X = sm.add_constant(data[x_col])
    model = sm.OLS(data[y_col], X).fit()
    intercept, slope = model.params['const'], model.params[x_col]
    r2 = model.rsquared
    x_grid = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    y_grid = intercept + slope * x_grid
    return slope, intercept, r2, x_grid, y_grid

# -------------------------------------------------------------------------
# Subplot 1: Age vs Charges
# -------------------------------------------------------------------------
sns.scatterplot(
    data=df, x='age', y='charges', hue='smoker', palette=hue_colors,
    alpha=0.6, ax=axes[0]
)

# Fits for Age
slope_age_s, int_age_s, r2_age_s, x_grid_a_s, y_grid_a_s = fit_ols_group(smoker_df, 'age')
slope_age_ns, int_age_ns, r2_age_ns, x_grid_a_ns, y_grid_a_ns = fit_ols_group(non_smoker_df, 'age')

# Plot regression lines
axes[0].plot(x_grid_a_s, y_grid_a_s, color='darkred', linestyle='--', linewidth=2.5,
             label=f'Smoker Fit: Y = {slope_age_s:.2f}X + {int_age_s:.2f} (R² = {r2_age_s:.3f})')
axes[0].plot(x_grid_a_ns, y_grid_a_ns, color='darkblue', linestyle='--', linewidth=2.5,
             label=f'Non-Smoker Fit: Y = {slope_age_ns:.2f}X + {int_age_ns:.2f} (R² = {r2_age_ns:.3f})')

axes[0].set_title("Age vs. Hospital Charges", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Age (years)", fontsize=11)
axes[0].set_ylabel("Hospital Charges ($)", fontsize=11)
axes[0].legend(title="Group & Linear Fit", frameon=True, fontsize=9)

print("Age vs Charges Regressions:")
print(f"  Smokers:     Slope = {slope_age_s:.4f}, Intercept = {int_age_s:.4f}, R² = {r2_age_s:.4f}")
print(f"  Non-Smokers: Slope = {slope_age_ns:.4f}, Intercept = {int_age_ns:.4f}, R² = {r2_age_ns:.4f}")

# -------------------------------------------------------------------------
# Subplot 2: BMI vs Charges
# -------------------------------------------------------------------------
sns.scatterplot(
    data=df, x='bmi', y='charges', hue='smoker', palette=hue_colors,
    alpha=0.6, ax=axes[1]
)

# Fits for BMI
slope_bmi_s, int_bmi_s, r2_bmi_s, x_grid_b_s, y_grid_b_s = fit_ols_group(smoker_df, 'bmi')
slope_bmi_ns, int_bmi_ns, r2_bmi_ns, x_grid_b_ns, y_grid_b_ns = fit_ols_group(non_smoker_df, 'bmi')

# Plot regression lines
axes[1].plot(x_grid_b_s, y_grid_b_s, color='darkred', linestyle='--', linewidth=2.5,
             label=f'Smoker Fit: Y = {slope_bmi_s:.2f}X + {int_bmi_s:.2f} (R² = {r2_bmi_s:.3f})')
axes[1].plot(x_grid_b_ns, y_grid_b_ns, color='darkblue', linestyle='--', linewidth=2.5,
             label=f'Non-Smoker Fit: Y = {slope_bmi_ns:.2f}X + {int_bmi_ns:.2f} (R² = {r2_bmi_ns:.3f})')

axes[1].set_title("BMI vs. Hospital Charges", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Body Mass Index (BMI)", fontsize=11)
axes[1].set_ylabel("Hospital Charges ($)", fontsize=11)
axes[1].legend(title="Group & Linear Fit", frameon=True, fontsize=9)

print("\nBMI vs Charges Regressions:")
print(f"  Smokers:     Slope = {slope_bmi_s:.4f}, Intercept = {int_bmi_s:.4f}, R² = {r2_bmi_s:.4f}")
print(f"  Non-Smokers: Slope = {slope_bmi_ns:.4f}, Intercept = {int_bmi_ns:.4f}, R² = {r2_bmi_ns:.4f}")

plt.suptitle("Problem 2(a): Patient Age and BMI vs Hospital Charges with Grouped Regression Lines", fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure file
fig_path = f"{FIGURES_DIR}/problem2_scatter_regressions.png"
plt.savefig(fig_path, dpi=300)

# Display inline in Google Colab output cell
plt.show()

print(f"Two-panel scatter plot saved to: {fig_path}")

print("\nConclusion (2a):")
print("- The bimodal distribution in medical hospital charges is primarily driven by smoking status and compounded by the BMI >= 30 obesity threshold.")
print("- Within smokers, non-obese individuals (BMI < 30) fall in the intermediate charge range ($15,000–$30,000), while obese smokers (BMI ≥ 30) occupy the high-charge range (>$30,000) due to greater cardiovascular, respiratory, and/or metabolic risks.")


# -------------------------------------------------------------------------
# PART (b): Logistic Regression Predicting Smoker Status
# -------------------------------------------------------------------------
print("\n--- Part (b): Logistic Regression (Predicting Smoker Status) ---")

X = df[['age', 'bmi', 'charges']]
y = (df['smoker'] == 'yes').astype(int)  # 1 = smoker, 0 = non-smoker

# Case 1: Same Data (100% / 100%)
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

# Case 2: 75% Train / 25% Test Split (random_state=10)
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