import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# ---------- Load artifact-free SPE data (first 1000 time points) ----------
hor_closed = np.load('vectores/spe_hor_closed_raw_wo.npy')[:, :1000]  # Horizontal, EC
hor_open   = np.load('vectores/spe_hor_open_raw_wo.npy')[:, :1000]    # Horizontal, EO
ver_closed = np.load('vectores/spe_ver_closed_raw_wo.npy')[:, :1000]  # Vertical, EC
ver_open   = np.load('vectores/spe_ver_open_raw_wo.npy')[:, :1000]    # Vertical, EO

# Time axis from 1 to 1000
time_points = np.arange(1, hor_open.shape[1] + 1)

# ---------- Function to compute average and standard deviation of SPE over time ----------
def compute_mean_over_time(data):
    means, stds = [], []
    for t in time_points:
        # For each subject, average SPE values from time 0 to t
        subject_means = np.mean(data[:, :t], axis=1)
        means.append(np.mean(subject_means))     # Mean over all subjects
        stds.append(np.std(subject_means))       # Std deviation across subjects
    return np.array(means), np.array(stds)

# ---------- Function to compute p-values at each time step ----------
def compute_temporal_pvalues(open_data, closed_data):
    pvalues = []
    for t in time_points:
        # For each subject, compute average up to time t for EO and EC
        data_open = np.mean(open_data[:, :t], axis=1)
        data_closed = np.mean(closed_data[:, :t], axis=1)
        pval = ttest_rel(data_open, data_closed).pvalue  # Paired t-test
        pvalues.append(pval)
    return np.array(pvalues)

# ---------- Compute metrics ----------
# Horizontal SPE
mean_hor_open, std_hor_open = compute_mean_over_time(hor_open)
mean_hor_closed, std_hor_closed = compute_mean_over_time(hor_closed)
pval_hor = compute_temporal_pvalues(hor_open, hor_closed)

# Vertical SPE
mean_ver_open, std_ver_open = compute_mean_over_time(ver_open)
mean_ver_closed, std_ver_closed = compute_mean_over_time(ver_closed)
pval_ver = compute_temporal_pvalues(ver_open, ver_closed)

# ---------- Create figure with two subplots ----------
fig, axs = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True)

# ---------- Plot a) Horizontal SPE ----------
ax1 = axs[0]
# Mean ± std for EO (blue)
lns1_1 = ax1.plot(time_points, mean_hor_open, label='Eyes Open', color='blue')
ax1.fill_between(time_points, mean_hor_open - std_hor_open, mean_hor_open + std_hor_open, alpha=0.3, color='blue')
# Mean ± std for EC (magenta)
lns1_2 = ax1.plot(time_points, mean_hor_closed, label='Eyes Closed', color='magenta')
ax1.fill_between(time_points, mean_hor_closed - std_hor_closed, mean_hor_closed + std_hor_closed, alpha=0.3, color='magenta')

# Y-axis settings
ax1.set_ylabel("Average SPE", fontsize=19)
ax1.tick_params(labelsize=19)
ax1.grid()

# Add secondary axis for p-values (log scale)
ax1r = ax1.twinx()
lns1_3 = ax1r.plot(time_points, pval_hor, color='black', label='p-value')
ax1r.set_yscale('log')
ax1r.set_ylabel("p-value", fontsize=19)
ax1r.tick_params(labelsize=19)

# ---------- Plot b) Vertical SPE ----------
ax2 = axs[1]
lns2_1 = ax2.plot(time_points, mean_ver_open, label='Eyes Open', color='blue')
ax2.fill_between(time_points, mean_ver_open - std_ver_open, mean_ver_open + std_ver_open, alpha=0.3, color='blue')

lns2_2 = ax2.plot(time_points, mean_ver_closed, label='Eyes Closed', color='magenta')
ax2.fill_between(time_points, mean_ver_closed - std_ver_closed, mean_ver_closed + std_ver_closed, alpha=0.3, color='magenta')

# Axis labels
ax2.set_xlabel("Analyzed time (sampling points)", fontsize=19)
ax2.set_ylabel("Average SPE", fontsize=19)
ax2.tick_params(labelsize=19)
ax2.grid()

# Add p-value curve on second y-axis (log scale)
ax2r = ax2.twinx()
lns2_3 = ax2r.plot(time_points, pval_ver, color='black', label='p-value')
ax2r.set_yscale('log')
ax2r.set_ylabel("p-value", fontsize=19)
ax2r.tick_params(labelsize=19)

# Subplot labels (a, b)
fig.text(-0.02, 0.91, 'a)', fontsize=19, fontweight='bold')
fig.text(-0.02, 0.43, 'b)', fontsize=19, fontweight='bold')

# Add legend (combined from both subplots)
lns = lns1_1 + lns1_2 + lns1_3
labels = [l.get_label() for l in lns]
fig.legend(lns, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=3, fontsize=18, frameon=False)

# Final layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Figura6_SPE_64.png", dpi=300, bbox_inches='tight')
plt.show()
