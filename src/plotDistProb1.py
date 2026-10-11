import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure target directories exist
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Load dataset
df = pd.read_csv("data/anemia.csv")
plt_data = df['PLT'].dropna()

# 2. Calculate summary statistics
mean_plt = plt_data.mean()
median_plt = plt_data.median()

# IQR Fencing Boundaries
q1, q3 = plt_data.quantile(0.25), plt_data.quantile(0.75)
iqr = q3 - q1
iqr_upper = q3 + 1.5 * iqr

# Modified Z-Score Boundaries (|ModZ| > 3.5)
mad = np.median(np.abs(plt_data - median_plt))
modz_upper = median_plt + (3.5 * mad / 0.6745)

# 3. Create Distribution Plot
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Histogram + KDE Curve
sns.histplot(plt_data, kde=True, color='teal', bins=30, stat="density", alpha=0.4)

# Mean vs. Median lines (Classic sign of right-skew / long tail)
plt.axvline(median_plt, color='blue', linestyle='--', linewidth=2, label=f'Median ({median_plt:.1f})')
plt.axvline(mean_plt, color='darkorange', linestyle='--', linewidth=2, label=f'Mean ({mean_plt:.1f})')

# Outlier Threshold Comparisons
plt.axvline(iqr_upper, color='crimson', linestyle=':', linewidth=2, label=f'IQR Fence (1.5xIQR = {iqr_upper:.1f})')
plt.axvline(modz_upper, color='purple', linestyle='-.', linewidth=2, label=f'ModZ Threshold (3.5 = {modz_upper:.1f})')

# Formatting & Annotation
plt.title("Platelet (PLT) Count Distribution: Right-Skewed / Long-Tailed Distribution", fontsize=13, fontweight='bold')
plt.xlabel("PLT Count (thousand cells/μL)", fontsize=11)
plt.ylabel("Density", fontsize=11)

# Annotate the long right tail
plt.annotate('Long Right Tail\n(Extreme Thrombocytosis)', 
             xy=(plt_data.max() - 10, 0.0005), 
             xytext=(plt_data.max() - 150, 0.0025),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

plt.legend(frameon=True, loc='upper right')
plt.tight_layout()

# Save figure
output_path = f"{FIGURES_DIR}/plt_long_tail_distribution.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Plot saved successfully to: {output_path}")