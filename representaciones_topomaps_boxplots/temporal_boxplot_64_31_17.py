# -------------------------------------------
# Script: temporal_boxplot_64_31_17.py
# Description: This script compares Permutation Entropy (PE) values between Eyes Closed (EC) and
#              Eyes Open (EO) conditions using 3 different electrode configurations: 64, 31, and 17 electrodes.
#              It creates a boxplot for each configuration, with statistical annotations to highlight
#              significant differences. The goal is to analyze how electrode reduction affects the ability
#              to detect differences between EC and EO.
# -------------------------------------------

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Function to create a boxplot with p-value annotation

def plot_boxplot_subplot(ax, ec_with, eo_with, ec_without, eo_without, ylabel, label_letra):
    data = [ec_with, eo_with, ec_without, eo_without]

    # Perform t-tests between EC and EO (with and without artifacts)
    t_stat_with, p_val_with = stats.ttest_ind(eo_with, ec_with, equal_var=False)
    t_stat_without, p_val_without = stats.ttest_ind(eo_without, ec_without, equal_var=False)

    # Box colors: pink for 'with', blue for 'without'
    colors = ['#FFB6C1', '#FFB6C1', '#ADD8E6', '#ADD8E6']

    # Create the boxplot
    box = ax.boxplot(data, patch_artist=True,
                     boxprops=dict(color='black'),
                     medianprops=dict(color='black'))

    # Fill boxes with colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Label axes
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["EC", "EO", "EC", "EO"], fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)

    # Adjust y-axis range with margin
    all_data = np.concatenate(data)
    y_max = np.max(all_data)
    y_min = np.min(all_data)
    margen_superior = (y_max - y_min) * 0.5
    margen_inferior = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margen_inferior, y_max + margen_superior)

    # Function to show significance using asterisks
    def get_p_asterisks(p):
        if p < 0.0001: return "****"
        elif p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        else: return "ns"

    # Draw statistical bars and annotations
    y_bar_with = y_max + margen_superior * 0.5
    y_palo_with = y_max + margen_superior * 0.3
    ax.plot([1, 1, 2, 2], [y_palo_with, y_bar_with, y_bar_with, y_palo_with], lw=2, color='black')
    ax.text(1.5, y_bar_with + (margen_superior * 0.05), get_p_asterisks(p_val_with),
            ha='center', va='bottom', color='black', fontsize=18, fontweight='bold')

    y_bar_without = y_max + margen_superior * 0.8
    y_palo_without = y_max + margen_superior * 0.6
    ax.plot([3, 3, 4, 4], [y_palo_without, y_bar_without, y_bar_without, y_palo_without], lw=2, color='black')
    ax.text(3.5, y_bar_without + (margen_superior * 0.05), get_p_asterisks(p_val_without),
            ha='center', va='bottom', color='black', fontsize=18, fontweight='bold')

    # Subplot label
    ax.text(-0.15, 1.05, label_letra, transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='top', ha='left')

# Load full data for PE (64 electrodes)
ec_with_pe = np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EC_PE_raw_4_1_w.npy")
eo_with_pe = np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EO_PE_raw_4_1_W.npy")
ec_without_pe = np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EC_PE_raw_4_1.npy")
eo_without_pe = np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EO_PE_raw_4_1.npy")

# Average PE across all 64 electrodes
ec_with_pe_64 = np.mean(ec_with_pe, axis=1)
eo_with_pe_64 = np.mean(eo_with_pe, axis=1)
ec_without_pe_64 = np.mean(ec_without_pe, axis=1)
eo_without_pe_64 = np.mean(eo_without_pe, axis=1)

# Average PE across 31 selected electrodes
new_order_31 = np.array([22, 23, 24, 30, 32, 34, 36, 38, 39,
                         2, 4, 6, 40, 43, 9, 11, 13, 44, 45,
                         16, 18, 20, 46, 56, 49, 51, 53, 55,
                         61, 62, 63]) - 1

ec_with_pe_31 = np.mean(ec_with_pe[:, new_order_31], axis=1)
eo_with_pe_31 = np.mean(eo_with_pe[:, new_order_31], axis=1)
ec_without_pe_31 = np.mean(ec_without_pe[:, new_order_31], axis=1)
eo_without_pe_31 = np.mean(eo_without_pe[:, new_order_31], axis=1)

# Average PE across 17 selected electrodes
new_order_17 = np.array([22, 23, 24, 32, 34, 36, 41, 9, 11, 13,
                         42, 49, 51, 53, 61, 62, 63]) - 1

ec_with_pe_17 = np.mean(ec_with_pe[:, new_order_17], axis=1)
eo_with_pe_17 = np.mean(eo_with_pe[:, new_order_17], axis=1)
ec_without_pe_17 = np.mean(ec_without_pe[:, new_order_17], axis=1)
eo_without_pe_17 = np.mean(eo_without_pe[:, new_order_17], axis=1)

# Create figure with 3 subplots (64, 31, and 17 electrodes)
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Boxplot for 64 electrodes
plot_boxplot_subplot(
    ax=axes[0],
    ec_with=ec_with_pe_64, eo_with=eo_with_pe_64,
    ec_without=ec_without_pe_64, eo_without=eo_without_pe_64,
    ylabel="PE",
    label_letra="a)"
)
axes[0].set_title("64 electrodes", fontsize=18, fontweight='bold')

# Boxplot for 31 electrodes
plot_boxplot_subplot(
    ax=axes[1],
    ec_with=ec_with_pe_31, eo_with=eo_with_pe_31,
    ec_without=ec_without_pe_31, eo_without=eo_without_pe_31,
    ylabel="PE",
    label_letra="b)"
)
axes[1].set_title("31 electrodes", fontsize=18, fontweight='bold')

# Boxplot for 17 electrodes
plot_boxplot_subplot(
    ax=axes[2],
    ec_with=ec_with_pe_17, eo_with=eo_with_pe_17,
    ec_without=ec_without_pe_17, eo_without=eo_without_pe_17,
    ylabel="PE",
    label_letra="c)"
)
axes[2].set_title("17 electrodes", fontsize=18, fontweight='bold')

# Final adjustments and save
plt.tight_layout()
plt.savefig("PE_64_vs_31_vs_17.png", dpi=300)
plt.show()
