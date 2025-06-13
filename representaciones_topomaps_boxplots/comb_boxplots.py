# -------------------------------------------
# Script: comb_boxplots.py
# Description: This script compares EEG features (Permutation Entropy, Skewness, and Kurtosis)
#              between Eyes Closed (EC) and Eyes Open (EO) conditions, both with and without artifacts.
#              It generates vertical boxplots showing the distribution for each condition and performs
#              statistical comparisons (t-tests) with asterisks indicating significance.
# -------------------------------------------

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Function to draw boxplots and show p-values for each pair

def plot_boxplot_subplot(ax, ec_with, eo_with, ec_without, eo_without, ylabel, label_letter):
    # Organize the data in the expected order
    data = [ec_with, eo_with, ec_without, eo_without]

    # Perform independent t-tests for both comparisons
    t_stat_with, p_val_with = stats.ttest_ind(eo_with, ec_with, equal_var=False)
    t_stat_without, p_val_without = stats.ttest_ind(eo_without, ec_without, equal_var=False)

    # Assign colors: pink = with artifacts, light blue = without artifacts
    colors = ['#FFB6C1', '#FFB6C1', '#ADD8E6', '#ADD8E6']

    # Draw the boxplots with black borders
    box = ax.boxplot(data, patch_artist=True,
        boxprops=dict(color='black', linewidth=2),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=2),
        capprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=6, linestyle='none')
    )

    # Fill boxes with the defined colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set x-axis labels and style
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["EC", "EO", "EC", "EO"], fontsize=36, fontweight='bold')
    ax.tick_params(axis='both', labelsize=36)

    # Calculate y-axis limits with margin
    all_data = np.concatenate(data)
    y_max = np.max(all_data)
    y_min = np.min(all_data)
    margin_top = (y_max - y_min) * 0.5
    margin_bottom = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin_bottom, y_max + margin_top)

    # Function to convert p-value into asterisk annotation
    def get_p_asterisks(p):
        if p < 0.0001: return "****"
        elif p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        else: return "ns"

    # Add annotation for p-value (with artifacts)
    y_bar_with = y_max + margin_top * 0.5
    y_line_with = y_max + margin_top * 0.3
    ax.plot([1, 1, 2, 2], [y_line_with, y_bar_with, y_bar_with, y_line_with], lw=2, color='black')
    ax.text(1.5, y_bar_with + (margin_top * 0.05), get_p_asterisks(p_val_with),
            ha='center', va='bottom', color='black', fontsize=36, fontweight='bold')

    # Add annotation for p-value (without artifacts)
    y_bar_without = y_max + margin_top * 0.8
    y_line_without = y_max + margin_top * 0.6
    ax.plot([3, 3, 4, 4], [y_line_without, y_bar_without, y_bar_without, y_line_without], lw=2, color='black')
    ax.text(3.5, y_bar_without + (margin_top * 0.05), get_p_asterisks(p_val_without),
            ha='center', va='bottom', color='black', fontsize=36, fontweight='bold')

    # Add subplot letter and vertical label
    ax.text(-0.15, 1.05, label_letter, transform=ax.transAxes,
            fontsize=40, fontweight='bold', va='top', ha='left')
    ax.text(-0.16, 0.5, rf"$\\langle {ylabel} \\rangle$", transform=ax.transAxes,
            fontsize=40, fontweight='bold', va='center', ha='center', rotation=90)

# ----------------------------
# Load data averaged by subject
# ----------------------------

# Load PE data
ec_with_pe = np.mean(np.load("MATRIX_FINAL_VALUES/EC_pe_beta_4_1_w.npy"), axis=1)
eo_with_pe = np.mean(np.load("MATRIX_FINAL_VALUES/EO_pe_beta_4_1_W.npy"), axis=1)
ec_without_pe = np.mean(np.load("MATRIX_FINAL_VALUES/EC_pe_beta_4_1_wo.npy"), axis=1)
eo_without_pe = np.mean(np.load("MATRIX_FINAL_VALUES/EO_pe_beta_4_1_Wo.npy"), axis=1)

# Load Skewness data
ec_with_skew = np.mean(np.load("MATRIX_FINAL_VALUES/EC_skew_beta_4_1_w.npy"), axis=1)
eo_with_skew = np.mean(np.load("MATRIX_FINAL_VALUES/EO_skew_beta_4_1_W.npy"), axis=1)
ec_without_skew = np.mean(np.load("MATRIX_FINAL_VALUES/EC_skew_beta_4_1_wo.npy"), axis=1)
eo_without_skew = np.mean(np.load("MATRIX_FINAL_VALUES/EO_skew_beta_4_1_Wo.npy"), axis=1)

# Load Kurtosis data
ec_with_kurt = np.mean(np.load("MATRIX_FINAL_VALUES/EC_kurt_beta_4_1_w.npy"), axis=1)
eo_with_kurt = np.mean(np.load("MATRIX_FINAL_VALUES/EO_kurt_beta_4_1_W.npy"), axis=1)
ec_without_kurt = np.mean(np.load("MATRIX_FINAL_VALUES/EC_kurt_beta_4_1_wo.npy"), axis=1)
eo_without_kurt = np.mean(np.load("MATRIX_FINAL_VALUES/EO_kurt_beta_4_1_Wo.npy"), axis=1)

# ----------------------------
# Create the final figure with 3 vertical subplots
# ----------------------------
fig, axes = plt.subplots(3, 1, figsize=(15, 30))

# PE boxplot (top)
plot_boxplot_subplot(axes[0], ec_with_pe, eo_with_pe, ec_without_pe, eo_without_pe, "PE", "a)")

# Skewness boxplot (middle)
plot_boxplot_subplot(axes[1], ec_with_skew, eo_with_skew, ec_without_skew, eo_without_skew, "S", "b)")

# Kurtosis boxplot (bottom)
plot_boxplot_subplot(axes[2], ec_with_kurt, eo_with_kurt, ec_without_kurt, eo_without_kurt, "K", "c)")

# Adjust and save figure
plt.tight_layout()
plt.subplots_adjust(left=0.22)
plt.savefig("PE_Skewness_Kurtosis_VERTICAL_FINAL_BLACKEDGES.png", dpi=300)
plt.show()
