"""
This script compares the Spatial Permutation Entropy (SPE) between two conditions: eyes open (EO) and eyes closed (EC), using EEG data. It looks at both horizontal and vertical configurations of the electrodes.

First, it loads the SPE data for each condition (EO/EC, horizontal/vertical). Then, it creates boxplots to show how the entropy values are distributed across the subjects in each case. I used different colors to separate the conditions more clearly.

After that, the code does a t-test to check if the differences between EO and EC are statistically significant, both in the horizontal and vertical directions. The results of the test are shown as asterisks on the plot (like * or ** depending on the p-value).

Finally, the figure is saved as a PNG file so I can include it in my report or use it for my presentation.

In general, this script helped me understand whether there are real differences in brain complexity (measured with SPE) depending on whether the eyes are open or closed, and whether that difference depends on how the electrodes are arranged.

"""


# Import necessary libraries
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# This script loads precomputed Spatial Permutation Entropy (SPE) results 
# from EEG signals for eyes open (EO) and eyes closed (EC) conditions.
# It performs statistical tests and generates a grouped boxplot to visualize 
# the differences between both conditions.

# ====== Load SPE data ======
# Horizontal configuration
spe_hor_closed = np.load("vectores/spe_hor_closed_raw_wo.npy")
spe_hor_open = np.load("vectores/spe_hor_open_raw_wo.npy")

# Vertical configuration
spe_ver_closed = np.load("vectores/spe_ver_closed_raw_wo.npy")
spe_ver_open = np.load("vectores/spe_ver_open_raw_wo.npy")

# ====== Define function to get p-value significance as stars ======
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
        return "ns"

# ====== Prepare figure and axes ======
fig, ax = plt.subplots(figsize=(7, 6))

# Combine all four data groups
all_data = [spe_hor_closed, spe_hor_open, spe_ver_closed, spe_ver_open]

# Set colors for horizontal (light pink) and vertical (light blue)
colors = ['#FFB6C1', '#FFB6C1', '#ADD8E6', '#ADD8E6']

# Create boxplot with black borders
bp = ax.boxplot(all_data, patch_artist=True, widths=0.4,
                medianprops=dict(color='black'),
                boxprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'))

# Apply facecolors
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Set x-axis labels and font
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(["EC", "EO", "EC", "EO"], fontsize=16, fontweight='bold')
ax.set_ylabel("SPE", fontsize=18, fontweight='bold')
ax.tick_params(axis='both', labelsize=16)

# ====== Add significance lines and asterisks ======
# Horizontal comparison
t_stat_hor, p_val_hor = stats.ttest_ind(spe_hor_open, spe_hor_closed, equal_var=False)
# Vertical comparison
t_stat_ver, p_val_ver = stats.ttest_ind(spe_ver_open, spe_ver_closed, equal_var=False)

# Y-axis range and padding
y_max = max([np.max(x) for x in all_data])
y_min = min([np.min(x) for x in all_data])
margin = (y_max - y_min) * 0.3
ax.set_ylim(y_min - margin * 0.3, y_max + margin)

# Draw bars and text for p-values
ax.plot([1, 1, 2, 2], [y_max, y_max + 0.05, y_max + 0.05, y_max], lw=2, color='black')
ax.text(1.5, y_max + 0.06, get_p_asterisks(p_val_hor), ha='center', va='bottom', fontsize=16)

ax.plot([3, 3, 4, 4], [y_max + 0.1, y_max + 0.15, y_max + 0.15, y_max + 0.1], lw=2, color='black')
ax.text(3.5, y_max + 0.16, get_p_asterisks(p_val_ver), ha='center', va='bottom', fontsize=16)

# ====== Save figure ======
plt.tight_layout()
plt.savefig("SPE_boxplot_with_pvalues.png", dpi=300)
plt.show()
