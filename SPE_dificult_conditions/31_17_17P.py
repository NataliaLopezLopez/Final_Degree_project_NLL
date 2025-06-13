# -------------------------------------------
# Script: 31_17_17P.py
# Description: This script visualises Spatial Permutation Entropy (SPE) values for Eyes Closed (EC)
#              and Eyes Open (EO) conditions, using two reduced electrode montages: 31 and 17 channels.
#              For each montage, it produces a vertical panel with two narrow boxplots – one for the
#              horizontal SPE and one for the vertical SPE. Independent t‑tests are used to compare EO
#              versus EC, and significance is marked with asterisks.
# -------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------------
# Helper: Convert p‑value into a star annotation
# --------------------------------------------------

def get_p_asterisks(p):
    """Return a string of asterisks that reflects the p‑value."""
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

# --------------------------------------------------
# 1) LOAD THE DATA  ────────────────────────────────
# Each .npy file has shape (subjects, channels).
# Horizontal (hor) and vertical (ver) SPE are stored separately.
# --------------------------------------------------

hc_31 = np.load("vectores_31montaje/spe_hor_closed_raw_wo_31.npy")  # EC horizontal, 31 ch
ho_31 = np.load("vectores_31montaje/spe_hor_open_raw_wo_31.npy")    # EO horizontal, 31 ch
vc_31 = np.load("vectores_31montaje/spe_ver_closed_raw_wo_31.npy")  # EC vertical, 31 ch
vo_31 = np.load("vectores_31montaje/spe_ver_open_raw_wo_31.npy")    # EO vertical, 31 ch

hc_17 = np.load("vectores_17montaje/spe_hor_closed_raw_wo_17.npy")  # EC horizontal, 17 ch
ho_17 = np.load("vectores_17montaje/spe_hor_open_raw_wo_17.npy")    # EO horizontal, 17 ch
vc_17 = np.load("vectores_17montaje/spe_ver_closed_raw_wo_17.npy")  # EC vertical, 17 ch
vo_17 = np.load("vectores_17montaje/spe_ver_open_raw_wo_17.npy")    # EO vertical, 17 ch

# --------------------------------------------------
# 2) PREPARE THE DATA  ─────────────────────────────
# We want one SPE value per subject, so we average across channels.
# The helper below first splits the array by channel and then stacks
# the means to keep the sample size consistent between montages.
# --------------------------------------------------

def split_and_stack(array: np.ndarray, n_elect: int) -> np.ndarray:
    """Split a 2‑D array into *n_elect* channel groups and stack them."""
    return np.hstack(np.array_split(array, n_elect))

# Pack information for the plotting loop
data_groups = [
    ("31 electrodes",
     [split_and_stack(hc_31, 31), split_and_stack(ho_31, 31)],  # Horizontal EC / EO
     [split_and_stack(vc_31, 31), split_and_stack(vo_31, 31)]), # Vertical EC / EO

    ("17 electrodes",
     [split_and_stack(hc_17, 17), split_and_stack(ho_17, 17)],  # Horizontal EC / EO
     [split_and_stack(vc_17, 17), split_and_stack(vo_17, 17)])  # Vertical EC / EO
]

# --------------------------------------------------
# 3) PLOT SET‑UP  ──────────────────────────────────
# Two rows (31 and 17 electrodes) × 1 column figure.
# --------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(6, 10))
subplot_letters = ["a)", "b)"]

for idx, (title, hor_data, ver_data) in enumerate(data_groups):
    ax = axes[idx]

    # Combine horizontal and vertical lists → 4 boxplots per panel
    all_data = hor_data + ver_data  # [EC‑hor, EO‑hor, EC‑ver, EO‑ver]
    positions = [1, 2, 4, 5]
    labels = ["EC", "EO", "EC", "EO"]

    # --------------------------------------------------
    # Draw boxplots
    # --------------------------------------------------
    bp = ax.boxplot(
        all_data,
        positions=positions,
        widths=0.4,
        patch_artist=True,
        medianprops=dict(color="black")
    )

    # Light‑blue fill for every box
    for patch in bp["boxes"]:
        patch.set_facecolor("#ADD8E6")

    # Axis styling
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=18, fontweight="bold")
    ax.tick_params(axis="y", labelsize=18)

    # Dynamic y‑axis range with margin
    y_max = max(np.max(d) for d in all_data)
    y_min = min(np.min(d) for d in all_data)
    margin = (y_max - y_min) * 0.2
    ax.set_ylim(y_min - margin, y_max + margin * 1.5)

    # --------------------------------------------------
    # Add statistical comparison bars (horizontal vs vertical)
    # --------------------------------------------------
    for j, (a_idx, b_idx) in enumerate([(1, 2), (4, 5)]):
        # Even indices: EC vs EO in horizontal (j=0) and vertical (j=1)
        t_stat, p_val = stats.ttest_ind(all_data[j*2], all_data[j*2 + 1], equal_var=False)
        y_line = y_max + margin * (0.1 + j * 0.2)
        y_bar = y_line + margin * 0.05
        ax.plot([a_idx, a_idx, b_idx, b_idx], [y_line, y_bar, y_bar, y_line], lw=2, color="black")
        ax.text((a_idx + b_idx) / 2, y_bar + margin * 0.02, get_p_asterisks(p_val),
                ha="center", va="bottom", fontsize=16, fontweight="bold")

    # Labels and annotations
    ax.set_ylabel("SPE", fontsize=20, fontweight="bold")
    ax.text(0.01, 1.05, subplot_letters[idx], transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top", ha="left")

    # Sub‑labels for horizontal and vertical columns
    ax.text(0.28, -0.12, "Horizontal", ha="center", va="top",
            fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(0.72, -0.12, "Vertical", ha="center", va="top",
            fontsize=18, fontweight="bold", transform=ax.transAxes)

# --------------------------------------------------
# 4) FINALISE AND SAVE  ────────────────────────────
# --------------------------------------------------

fig.align_ylabels(axes)
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig("SPE_31_17_vertical_alineado.png", format="png", dpi=1200, bbox_inches="tight")
plt.show()
