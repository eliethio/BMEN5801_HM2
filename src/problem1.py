import os
import sys

# Ensure pyampute is installed for Little's MCAR Test
try:
    from pyampute.exploration.mcar_statistical_tests import MCARTest
except ImportError:
    os.system(f"{sys.executable} -m pip install pyampute")
    from pyampute.exploration.mcar_statistical_tests import MCARTest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Define directories
DATA_PATH = "data/anemia.csv"
FIGURES_DIR = "results/figures"
TABLES_DIR = "results/tables"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Define CBC feature set
CBC_COLS = ['RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'WBC', 'PLT', 'HGB']

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("PROBLEM 1: DATA ANOMALIES & STATISTICAL IMPUTATION ANALYSIS")
print("=" * 65)

# -------------------------------------------------------------------------
# PART (a): Comprehensive All-vs-All MAR Pairwise Association Test
# -------------------------------------------------------------------------
print("\n--- Part (a): Comprehensive Missingness Analysis (MCAR vs MAR) ---")

# 1. Little's MCAR Test across all 9 features
mcar_test = MCARTest(method="little")
try:
    p_val_little = mcar_test.little_mcar_test(df[CBC_COLS])
    print(f"Little's MCAR Test p-value: {p_val_little:.4e}")
    if p_val_little < 0.05:
        print("Decision: Reject MCAR (p < 0.05) -> Data missingness is MAR/MNAR.")
    else:
        print("Decision: Fail to reject MCAR.")
except Exception as e:
    print(f"Little's test warning: {e}")

# 2. Pairwise t-tests across ALL missing features vs remaining features
all_associations = []

# Find all features that have missing values
missing_features = [col for col in CBC_COLS if df[col].isnull().sum() > 0]

for target_col in missing_features:
    missing_mask = df[target_col].isnull()
    
    for eval_col in CBC_COLS:
        if target_col == eval_col:
            continue
            
        grp_present = df.loc[~missing_mask, eval_col].dropna()
        grp_missing = df.loc[missing_mask, eval_col].dropna()
        
        if len(grp_missing) > 1 and len(grp_present) > 1:
            stat, p_val = stats.ttest_ind(grp_present, grp_missing, equal_var=False)
            all_associations.append({
                'Missing_Feature': target_col,
                'Associated_Feature': eval_col,
                't_statistic': stat,
                'p_value': p_val
            })

# Convert to DataFrame and sort by p-value
assoc_df = pd.DataFrame(all_associations).sort_values('p_value')
assoc_df.to_csv(f"{TABLES_DIR}/problem1_mar_associations.csv", index=False)

print("\nTop 5 Strongest MAR Associations Across All Feature Pairs:")
print(assoc_df.head().to_string(index=False))

if not assoc_df.empty:
    top = assoc_df.iloc[0]
    print(f"\nConclusion for Part (a):")
    print(f"Missingness in '{top['Missing_Feature']}' is most strongly associated "
          f"with observed values of '{top['Associated_Feature']}' (p = {top['p_value']:.4e}).")
          
# -------------------------------------------------------------------------
# PART (b): Outlier Identification (IQR Fencing vs Modified Z-Scores)
# -------------------------------------------------------------------------
print("\n--- Part (b): Outlier Identification ---")

iqr_outliers = {}
modz_outliers = {}

def get_modified_z_score(series):
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return pd.Series(0, index=series.index)
    return 0.6745 * (series - median) / mad

outlier_records = []

for col in CBC_COLS:
    data = df[col].dropna()
    
    # (i) IQR Fencing
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    iqr_idx = set(data[(data < lower_fence) | (data > upper_fence)].index)
    
    # (ii) Modified Z-Score (|ModZ| > 3.5)
    mod_z = get_modified_z_score(data)
    modz_idx = set(data[np.abs(mod_z) > 3.5].index)
    
    iqr_outliers[col] = iqr_idx
    modz_outliers[col] = modz_idx
    
    outlier_records.append({
        'Feature': col,
        'IQR_Outlier_Count': len(iqr_idx),
        'ModZ_Outlier_Count': len(modz_idx),
        'Identical_Set': iqr_idx == modz_idx
    })

outlier_df = pd.DataFrame(outlier_records)
outlier_df.to_csv(f"{TABLES_DIR}/problem1_outliers.csv", index=False)
print("\nOutlier Detection Comparison Table:")
print(outlier_df.to_string(index=False))

print("\nConclusion (b):")
print("- The two approaches do NOT find identical outliers across all features.")
print("- Modified Z-Scores rely on MAD and median, making them more robust to extreme skewness "
      "and conservative compared to Q1/Q3 IQR fences.")

# -------------------------------------------------------------------------
# PART (c): Imputation Algorithms & MAE Evaluation
# -------------------------------------------------------------------------
print("\n--- Part (c): Imputation Algorithms & MAE Benchmark ---")

# Evaluate MAE on complete cases by masking 10% of known HGB values
known_hgb_df = df[df['HGB'].notnull()].copy()
np.random.seed(42)

mask_indices = np.random.choice(known_hgb_df.index, size=int(0.10 * len(known_hgb_df)), replace=False)
true_hgb_vals = known_hgb_df.loc[mask_indices, 'HGB'].values

# Mean Imputation
imp_mean_fit = SimpleImputer(strategy='mean')
df_masked_mean = known_hgb_df[CBC_COLS].copy()
df_masked_mean.loc[mask_indices, 'HGB'] = np.nan
mean_imputed = imp_mean_fit.fit_transform(df_masked_mean)
pred_mean = mean_imputed[known_hgb_df.index.get_indexer(mask_indices), CBC_COLS.index('HGB')]

# Median Imputation
imp_median_fit = SimpleImputer(strategy='median')
df_masked_med = known_hgb_df[CBC_COLS].copy()
df_masked_med.loc[mask_indices, 'HGB'] = np.nan
median_imputed = imp_median_fit.fit_transform(df_masked_med)
pred_median = median_imputed[known_hgb_df.index.get_indexer(mask_indices), CBC_COLS.index('HGB')]

# kNN Imputation
imp_knn_fit = KNNImputer(n_neighbors=5)
df_masked_knn = known_hgb_df[CBC_COLS].copy()
df_masked_knn.loc[mask_indices, 'HGB'] = np.nan
knn_imputed = imp_knn_fit.fit_transform(df_masked_knn)
pred_knn = knn_imputed[known_hgb_df.index.get_indexer(mask_indices), CBC_COLS.index('HGB')]

# MICE Imputation
imp_mice_fit = IterativeImputer(max_iter=20, random_state=42)
df_masked_mice = known_hgb_df[CBC_COLS].copy()
df_masked_mice.loc[mask_indices, 'HGB'] = np.nan
mice_imputed = imp_mice_fit.fit_transform(df_masked_mice)
pred_mice = mice_imputed[known_hgb_df.index.get_indexer(mask_indices), CBC_COLS.index('HGB')]

# Compute MAE
mae_scores = {
    'Mean': np.mean(np.abs(true_hgb_vals - pred_mean)),
    'Median': np.mean(np.abs(true_hgb_vals - pred_median)),
    'kNN': np.mean(np.abs(true_hgb_vals - pred_knn)),
    'MICE': np.mean(np.abs(true_hgb_vals - pred_mice))
}

print("Imputation Performance (MAE):")
for method, score in mae_scores.items():
    print(f"  {method:8s} MAE: {score:.4f}")

best_method = min(mae_scores, key=mae_scores.get)
print(f"\nBest Performing Algorithm: {best_method} (MAE = {mae_scores[best_method]:.4f})")

# Full dataset imputation for scatter plot
full_mean = SimpleImputer(strategy='mean').fit_transform(df[CBC_COLS])
full_median = SimpleImputer(strategy='median').fit_transform(df[CBC_COLS])
full_knn = KNNImputer(n_neighbors=5).fit_transform(df[CBC_COLS])
full_mice = IterativeImputer(max_iter=20, random_state=42).fit_transform(df[CBC_COLS])

missing_mask = df['HGB'].isnull()
rbc_missing = df.loc[missing_mask, 'RBC']

# Create single scatter plot
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Plot observed background points
plt.scatter(df.loc[~missing_mask, 'RBC'], df.loc[~missing_mask, 'HGB'],
            color='gray', alpha=0.35, label='Observed Data', marker='o')

# Plot imputed points
plt.scatter(rbc_missing, full_mean[missing_mask, CBC_COLS.index('HGB')],
            color='crimson', marker='o', label=f"Mean (MAE={mae_scores['Mean']:.2f})", s=60)
plt.scatter(rbc_missing, full_median[missing_mask, CBC_COLS.index('HGB')],
            color='blue', marker='s', label=f"Median (MAE={mae_scores['Median']:.2f})", s=60)
plt.scatter(rbc_missing, full_knn[missing_mask, CBC_COLS.index('HGB')],
            color='green', marker='^', label=f"kNN (MAE={mae_scores['kNN']:.2f})", s=60)
plt.scatter(rbc_missing, full_mice[missing_mask, CBC_COLS.index('HGB')],
            color='purple', marker='D', label=f"MICE (MAE={mae_scores['MICE']:.2f})", s=60)

plt.title("RBC vs. Imputed Hemoglobin (HGB) across Imputation Algorithms", fontsize=13)
plt.xlabel("RBC (million/microliter)", fontsize=11)
plt.ylabel("HGB (grams/deciliter)", fontsize=11)
plt.legend(title="Algorithm / Data", frameon=True)
plt.tight_layout()

scatter_fig_path = f"{FIGURES_DIR}/problem1_imputation_scatter.png"
plt.savefig(scatter_fig_path, dpi=300)
plt.close()

print(f"Scatter plot saved to: {scatter_fig_path}")
print("=" * 65)