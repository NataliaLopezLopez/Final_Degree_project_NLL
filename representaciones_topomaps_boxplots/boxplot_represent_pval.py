# -------------------------------------------
# Script: boxplot_represent_pval.py
# Description: This script compares the EEG features Permutation Entropy (PE), Skewness, and Kurtosis
#              between Eyes Closed (EC) and Eyes Open (EO) conditions. The data used has been cleaned
#              to remove artifacts. For each feature, a boxplot is created and a statistical comparison
#              (t-test) is performed to determine if the difference between EC and EO is significant.
#              Asterisks are added to indicate significance level.
# -------------------------------------------

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Function to draw a boxplot with the p-value comparison between EC and EO

def plot_boxplot_subplot(ax, ec_data, eo_data, ylabel, label_letter):
    # Perform an independent t-test between EO and EC
    t_stat, p_val = stats.ttest_ind(eo_data, ec_data, equal_var=False)

    # This function returns a string of asterisks depending on the p-value
    def get_p_asterisks(p):
        if p < 0.0001:
            return "****"
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"  # not significant

    # Colors for the boxplots
    colors = ['#ADD8E6', '#ADD8E6']

    # Create the boxplot
    box = ax.boxplot([ec_data, eo_data], patch_artist=True,
                     boxprops=dict(color='black'),
                     medianprops=dict(color='black'))

    # Fill the boxes with the selected colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set axis labels and tick styles
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["EC", "EO"], fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)

    # Set y-axis limits with some margin
    all_data = np.concatenate([ec_data, eo_data])
    y_max = np.max(all_data)
    y_min = np.min(all_data)
    margin_top = (y_max - y_min) * 0.4
    margin_bottom = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin_bottom, y_max + margin_top)

    # Draw the bar and p-value on top of the plot
    y_bar = y_max + margin_top * 0.5
    y_line = y_max + margin_top * 0.3
    ax.plot([1, 1, 2, 2], [y_line, y_bar, y_bar, y_line], lw=2, color='black')
    ax.text(1.5, y_bar + (margin_top * 0.05), get_p_asterisks(p_val),
            ha='center', va='bottom', color='black', fontsize=20, fontweight='bold')

    # Add label for each subplot
    ax.text(-0.15, 1.08, label_letter, transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='top', ha='left')

# ---------------------------------------------------------------
# Load feature data averaged per subject (Alpha band, no artifacts)
# ---------------------------------------------------------------

# Load PE data
ec_without_pe = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EC_PE_ALPHA_4_1_W_59.npy"), axis=1)
eo_without_pe = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EO_PE_ALPHA_4_1_W_59.npy"), axis=1)

# Load Skewness data
ec_without_skew = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EC_skewness_ALPHA_4_1_W_59.npy"), axis=1)
eo_without_skew = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EO_skewness_ALPHA_4_1_W_59.npy"), axis=1)

# Load Kurtosis data
ec_without_kurt = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EC_kurtosis_ALPHA_4_1_W_59.npy"), axis=1)
eo_without_kurt = np.mean(np.load("tabla_pe_skew_kurt_raw_rawwo_bewo/EO_kurtosis_ALPHA_4_1_W_59.npy"), axis=1)

# ---------------------------------------------------------------
# Create figure with 3 boxplots (one for each metric)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot PE (Permutation Entropy)
plot_boxplot_subplot(
    ax=axes[0],
    ec_data=ec_without_pe,
    eo_data=eo_without_pe,
    ylabel="PE",
    label_letter="a)"
)

# Plot Skewness
plot_boxplot_subplot(
    ax=axes[1],
    ec_data=ec_without_skew,
    eo_data=eo_without_skew,
    ylabel="Skewness",
    label_letter="b)"
)

# Plot Kurtosis
plot_boxplot_subplot(
    ax=axes[2],
    ec_data=ec_without_kurt,
    eo_data=eo_without_kurt,
    ylabel="Kurtosis",
    label_letter="c)"
)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("PE_Skewness_Kurtosis_alpha_WITHOUT.png", dpi=300)
plt.show()
