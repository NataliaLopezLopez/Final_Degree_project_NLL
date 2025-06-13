from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Load horizontal and vertical SPE data (with and without artifacts)
spe_hor_closed_wo = np.load("vectores/primeros_vect/spe_hor_closed_raw_wo.npy")
spe_hor_open_wo = np.load("vectores/primeros_vect/spe_hor_open_raw_wo.npy")
spe_hor_closed_w = np.load("vectores/primeros_vect/spe_hor_closed_raw_w.npy")
spe_hor_open_w = np.load("vectores/primeros_vect/spe_hor_open_raw_w.npy")

spe_ver_closed_wo = np.load("vectores/primeros_vect/spe_ver_closed_raw_wo.npy")
spe_ver_open_wo = np.load("vectores/primeros_vect/spe_ver_open_raw_wo.npy")
spe_ver_closed_w = np.load("vectores/primeros_vect/spe_ver_closed_raw_w.npy")
spe_ver_open_w = np.load("vectores/primeros_vect/spe_ver_open_raw_w.npy")

# Function to turn p-values into significance stars
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

# Create vertical figure with 2 boxplot panels (horizontal and vertical)
fig, axs = plt.subplots(2, 1, figsize=(6, 10))

# Define color scheme: pink for 'with artifacts', blue for 'without'
color_dict = {0: '#FFB6C1', 1: '#FFB6C1', 2: '#ADD8E6', 3: '#ADD8E6'}

# Function to draw boxplots and annotate statistics
def plot_raw(ax, data, orientation, letter):
    # Draw the boxplot with four groups
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color='black'), widths=0.4)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(color_dict[i])

    # Set x labels and styles
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["EC", "EO", "EC", "EO"], fontsize=18, fontweight='bold')
    ax.tick_params(labelsize=18)
    ax.text(0.05, 1.05, letter, transform=ax.transAxes, size=20, weight='bold')

    # Define y-axis range
    y_max = max([np.max(x) for x in data])
    y_min = min([np.min(x) for x in data])
    margen = (y_max - y_min) * 0.2
    ax.set_ylim(y_min - margen, y_max + margen)

    # Add p-value annotations between each EC/EO pair
    for i, (a, b) in enumerate([(0, 1), (2, 3)]):
        t_stat, p_val = stats.ttest_ind(data[a], data[b], equal_var=False)
        y_bar = y_max + margen * (0.1 + i * 0.2)
        y_line = y_bar - margen * 0.05
        ax.plot([a+1, a+1, b+1, b+1], [y_line, y_bar, y_bar, y_line], lw=2, color='black')
        ax.text((a + b)/2 + 1, y_bar + margen * 0.02, get_p_asterisks(p_val),
                ha='center', va='bottom', fontsize=16, fontweight='bold')

# Call function for horizontal SPE
plot_raw(axs[0],
         [spe_hor_closed_w, spe_hor_open_w, spe_hor_closed_wo, spe_hor_open_wo],
         "Horizontal", "a)")

# Call function for vertical SPE
plot_raw(axs[1],
         [spe_ver_closed_w, spe_ver_open_w, spe_ver_closed_wo, spe_ver_open_wo],
         "Vertical", "b)")

# Adjust layout and save the figure
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig("boxplot_spatial_horizontal_vertical.png", dpi=300)
plt.show()
