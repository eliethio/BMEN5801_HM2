import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Setup Output Directories
DATA_PATH = "data/sleeplifestyle.csv"
FIGURES_DIR = "results/figures"
TABLES_DIR = "results/tables"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 65)
print("PROBLEM 3: MULTINOMIAL LOGISTIC REGRESSION WITH REGULARIZATION")
print("=" * 65)

# 1. Load Dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'. Please ensure 'sleeplifestyle.csv' is in 'data/'.")

df = pd.read_csv(DATA_PATH)

# Drop PersonID / Person ID column
id_cols = [c for c in df.columns if 'person' in c.lower() and 'id' in c.lower()]
if id_cols:
    df = df.drop(columns=id_cols)

# Fill NaN values in 'Sleep Disorder' target with 'Regular Sleep'
target_col = 'Sleep Disorder'
df[target_col] = df[target_col].fillna('Regular Sleep').replace({'None': 'Regular Sleep'})

# Parse 'Blood Pressure' into 'Systolic Pressure' and 'Diastolic Pressure'
if 'Blood Pressure' in df.columns:
    df[['Systolic Pressure', 'Diastolic Pressure']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    df = df.drop(columns=['Blood Pressure'])

# One-Hot Encode Categorical Features (Gender, BMI Category)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols.remove(target_col)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]
classes = np.unique(y)

# 2. Train / Test Split (75% Train, 25% Test, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10, stratify=y
)

# Standardize Features for SAGA Solver Numerical Stability
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Define the 4 Models using Unified 'saga' Solver
sol = 'saga'
models = {
    "(i) No Regularization": LogisticRegression(penalty=None, solver=sol, random_state=10, max_iter=5000),
    "(ii) Ridge (L2)": LogisticRegression(penalty='l2', C=1.0, solver=sol, random_state=10, max_iter=5000),
    "(iii) Lasso (L1)": LogisticRegression(penalty='l1', C=1.0, solver=sol, random_state=10, max_iter=5000),
    "(iv) Elastic Net (L1 ratio = 0.5)": LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=1.0, solver=sol, random_state=10, max_iter=5000)
}

# 4. Fit Models, Plot Side-by-Side Confusion Matrices, and Save CSV Reports
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    acc = model.score(X_test_scaled, y_test)
    
    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False, values_format='d')
    axes[idx].set_title(f"{name}\nAccuracy: {acc*100:.2f}%", fontsize=11, fontweight='bold')
    
    # Classification Report Export (Formatted to 4 Sig Figs)
    cr_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    df_cr = pd.DataFrame(cr_dict).transpose()
    df_cr_formatted = df_cr.applymap(lambda x: float(f"{x:.4g}") if isinstance(x, (float, int)) else x)
    
    table_path = f"{TABLES_DIR}/problem3_classification_report_model_{idx+1}.csv"
    df_cr_formatted.to_csv(table_path)

plt.suptitle("Problem 3: Multinomial Logistic Regression Confusion Matrices Across Regularization Methods", fontsize=14, fontweight='bold')
plt.tight_layout()

fig_path = f"{FIGURES_DIR}/problem3_confusion_matrices.png"
plt.savefig(fig_path, dpi=300)
plt.show()

print(f"Plot saved to: {fig_path}")