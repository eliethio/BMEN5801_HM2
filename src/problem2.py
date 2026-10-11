import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Try importing IPython display for seamless Colab inline rendering
try:
    from IPython.display import display, Image
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

# Setup Directories
DATA_PATH = "data/insurance.csv"
FIGURES_DIR = "results/figures"
TABLES_DIR = "results/tables"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# 1. Load Dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'. Please ensure 'insurance.csv' is in the 'data/' folder.")

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

def fit_ols_group(data, x_col, y_col='charges'):
    X = sm.add_constant(data[x_col])
    model = sm.OLS(data[y_col], X).fit()
    intercept, slope = model.params['const'], model.params[x_col]
    r2 = model.rsquared
    x_grid = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    y_grid = intercept + slope * x_grid
    return slope, intercept, r2, x_grid, y_grid

# Subplot 1: Age vs Charges
sns.scatterplot(
    data=df, x='age', y='charges', hue='smoker', palette=hue_colors,
    alpha=0.6, ax=axes[0]
)

slope_age_s, int_age_s, r2_age_s, x_grid_a_s, y_grid_a_s = fit_ols_group(smoker_df, 'age')
slope_age_ns, int_age_ns, r2_age_ns, x_grid_a_ns, y_grid_a_ns = fit_ols_group(non_smoker_df, 'age')

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

# Subplot 2: BMI vs Charges
sns.scatterplot(
    data=df, x='bmi', y='charges', hue='smoker', palette=hue_colors,
    alpha=0.6, ax=axes[1]
)

slope_bmi_s, int_bmi_s, r2_bmi_s, x_grid_b_s, y_grid_b_s = fit_ols_group(smoker_df, 'bmi')
slope_bmi_ns, int_bmi_ns, r2_bmi_ns, x_grid_b_ns, y_grid_b_ns = fit_ols_group(non_smoker_df, 'bmi')

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

fig_path_a = f"{FIGURES_DIR}/problem2_scatter_regressions.png"
plt.savefig(fig_path_a, dpi=300)
plt.show()

if HAS_IPYTHON:
    display(Image(filename=fig_path_a))

print(f"Two-panel scatter plot saved to: {fig_path_a}")

# -------------------------------------------------------------------------
# Part (a) Discussion: Bimodal Distribution Drivers
# -------------------------------------------------------------------------
print("\n" + "-" * 65)
print("PART (a) DISCUSSION: BIMODAL DISTRIBUTION ANALYSIS")
print("-" * 65)
print("""
What factor(s) explain the bimodal distribution in hospital charges?

1. Primary Macro Driver - Smoking Status:
   Smoking status creates the primary base separation in hospital charges.
   Non-smokers form a unimodal baseline distribution clustered at lower charges
   (mostly below $15,000). Smokers experience an overall upward shift in baseline
   medical costs across all age groups.

2. Secondary Interaction Driver - Obesity Threshold (BMI >= 30):
   Within the smoker population, charges split into two distinct sub-modes:
     - Non-obese smokers (BMI < 30): Incur moderately elevated charges ($15,000 - $30,000).
     - Obese smokers (BMI >= 30): Experience a drastic, multiplicative jump in healthcare
       costs ($30,000 - $60,000) due to severe combined cardiovascular and metabolic risk.
   Thus, the non-linear interaction between smoking status and obesity creates the
   upper mode in hospital charges.
""")
print("-" * 65)

# -------------------------------------------------------------------------
# PART (b): Logistic Regression Predicting Smoker Status
# -------------------------------------------------------------------------
print("\n--- Part (b): Logistic Regression (Predicting Smoker Status) ---")

X = df[['age', 'bmi', 'charges']]
y = (df['smoker'] == 'yes').astype(int)  # 1 = smoker, 0 = non-smoker

# Case 1: Training and Testing on FULL Dataset (100% / 100%)
print("\n[Case 1]: Training and Testing on FULL Dataset (100% / 100%)")
clf_same = LogisticRegression(C=1.0, random_state=10, max_iter=1000)
#clf_same = LogisticRegression(C=100, random_state=10, max_iter=1000)
clf_same.fit(X, y)
y_pred_same = clf_same.predict(X)

cm_same = confusion_matrix(y, y_pred_same)
cr_same_dict = classification_report(y, y_pred_same, target_names=['Non-Smoker', 'Smoker'], output_dict=True)
cr_same_str = classification_report(y, y_pred_same, target_names=['Non-Smoker', 'Smoker'], digits=4)
acc_same = clf_same.score(X, y)

print("Confusion Matrix (Case 1):")
print(pd.DataFrame(cm_same, index=['True Non-Smoker', 'True Smoker'], columns=['Pred Non-Smoker', 'Pred Smoker']))
print("\nClassification Report (Case 1):")
print(cr_same_str)
print(f"Overall Accuracy (Same Data): {acc_same * 100:.2f}%")

# Case 2: 75% Train / 25% Test Split (random_state=10)
print("\n[Case 2]: Train on 75% / Test on 25% Split (random_state=10)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10, stratify=y)

clf_split = LogisticRegression(C=1.0, random_state=10, max_iter=1000)

#clf_split = LogisticRegression(C=100, random_state=10, max_iter=1000)
clf_split.fit(X_train, y_train)
y_pred_split = clf_split.predict(X_test)

cm_split = confusion_matrix(y_test, y_pred_split)
cr_split_dict = classification_report(y_test, y_pred_split, target_names=['Non-Smoker', 'Smoker'], output_dict=True)
cr_split_str = classification_report(y_test, y_pred_split, target_names=['Non-Smoker', 'Smoker'], digits=4)
acc_split = clf_split.score(X_test, y_test)

print("Confusion Matrix (Case 2 - Test Set):")
print(pd.DataFrame(cm_split, index=['True Non-Smoker', 'True Smoker'], columns=['Pred Non-Smoker', 'Pred Smoker']))
print("\nClassification Report (Case 2 - Test Set):")
print(cr_split_str)
print(f"Overall Accuracy (75/25 Split): {acc_split * 100:.2f}%")

# -------------------------------------------------------------------------
# Generate and Save Confusion Matrix Plots for Both Cases
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Case 1 Confusion Matrix
disp_same = ConfusionMatrixDisplay(confusion_matrix=cm_same, display_labels=['Non-Smoker', 'Smoker'])
disp_same.plot(ax=axes[0], cmap='Blues', values_format='d', colorbar=False)
axes[0].set_title(f"Case 1: Full Dataset (100% / 100%)\nAccuracy: {acc_same*100:.2f}%", fontsize=12, fontweight='bold')

# Plot Case 2 Confusion Matrix
disp_split = ConfusionMatrixDisplay(confusion_matrix=cm_split, display_labels=['Non-Smoker', 'Smoker'])
disp_split.plot(ax=axes[1], cmap='Greens', values_format='d', colorbar=False)
axes[1].set_title(f"Case 2: 75% Train / 25% Test Split\nAccuracy: {acc_split*100:.2f}%", fontsize=12, fontweight='bold')

plt.suptitle("Problem 2(b): Logistic Regression Confusion Matrices", fontsize=14, fontweight='bold')
plt.tight_layout()

cm_fig_path = f"{FIGURES_DIR}/problem2_confusion_matrices.png"
plt.savefig(cm_fig_path, dpi=300)
plt.show()

if HAS_IPYTHON:
    display(Image(filename=cm_fig_path))

print(f"Confusion matrix plot saved to: {cm_fig_path}")

# -------------------------------------------------------------------------
# Save Classification Reports as CSV Tables (Rounded to 4 Significant Digits)
# -------------------------------------------------------------------------
df_cr_same = pd.DataFrame(cr_same_dict).transpose()
df_cr_split = pd.DataFrame(cr_split_dict).transpose()

# Format all float metrics to 4 significant digits
df_cr_same = df_cr_same.applymap(lambda x: float(f"{x:.4g}") if isinstance(x, (float, int)) else x)
df_cr_split = df_cr_split.applymap(lambda x: float(f"{x:.4g}") if isinstance(x, (float, int)) else x)

table_same_path = f"{TABLES_DIR}/problem2_classification_report_100pct.csv"
table_split_path = f"{TABLES_DIR}/problem2_classification_report_75_25.csv"

df_cr_same.to_csv(table_same_path)
df_cr_split.to_csv(table_split_path)

print(f"Classification report (Case 1) saved to: {table_same_path} (4 sig figs)")
print(f"Classification report (Case 2) saved to: {table_split_path} (4 sig figs)")

print("=" * 65)