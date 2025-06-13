# -------------------------------------------
# Script: boxplot_p_val_generardato.py
# Description: This script calculates Permutation Entropy (PE) for Eyes Open (EO) and Eyes Closed (EC)
#              EEG conditions using a custom `eeg` class. The computation is parallelised with
#              `multiprocess` to speed up per‑subject calculations (109 subjects). Once PE values are
#              obtained, the script performs an independent t‑test (EO vs EC) and visualises the
#              distribution with a simple boxplot. The figure lets you quickly see whether PE differs
#              significantly between brain states.
# -------------------------------------------

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar for loops
import multiprocess as mp
from datetime import datetime
from egg_utils_2 import eeg  # custom class for EEG handling

# --------------------------------------------------
# 1) ANALYSIS PARAMETERS  ──────────────────────────
# --------------------------------------------------

N_SUBJECTS = 109      # Total number of participants
FILT_MODE = "raw"     # Use raw (unfiltered) data
WORD_LENGTH = 3       # Pattern length for PE
LAG = 1               # Lag between samples in a pattern

# Instantiate EEG objects for EO (run=1) and EC (run=2)
eeg_open = eeg(N_SUBJECTS, FILT_MODE, run=1)
eeg_closed = eeg(N_SUBJECTS, FILT_MODE, run=2)

# --------------------------------------------------
# 2) SET COMMON ATTRIBUTES  ────────────────────────
# --------------------------------------------------

for eeg_obj in (eeg_open, eeg_closed):
    eeg_obj.L = WORD_LENGTH
    eeg_obj.lag = LAG
    eeg_obj.cut_up = 30   # Upper frequency bound (Hz)
    eeg_obj.cut_low = 12  # Lower frequency bound (Hz)
    eeg_obj.file_path = "/Users/natalialopezlopezicloud.com/Desktop/Escritorio2/GAIA/eeg-spatial-analysis-main/files-2"

# --------------------------------------------------
# 3) LOAD DATA  ────────────────────────────────────
# --------------------------------------------------

eeg_open.load_data()
eeg_closed.load_data()

# --------------------------------------------------
# 4) COMPUTE PERMUTATION ENTROPY (PE)  ────────────
# --------------------------------------------------

start = datetime.now()

def compute_pe(eeg_obj):
    """Helper to compute PE for a single subject index."""
    return [eeg_obj.PE_chanel(sub_idx) for sub_idx in range(eeg_obj.subjects)]

if __name__ == "__main__":
    with mp.Pool(mp.cpu_count()) as pool:
        # EO PE
        print("→ Calculating PE for Eyes Open (EO)…")
        pe_eyes_open = pool.map(eeg_open.PE_chanel, range(eeg_open.subjects))

        # EC PE
        print("→ Calculating PE for Eyes Closed (EC)…")
        pe_eyes_closed = pool.map(eeg_closed.PE_chanel, range(eeg_closed.subjects))

print("PE computation completed in", datetime.now() - start)

# Convert lists to NumPy arrays for easier math
pe_eyes_open = np.stack(pe_eyes_open)
pe_eyes_closed = np.stack(pe_eyes_closed)

# --------------------------------------------------
# 5) STATISTICAL TEST  ─────────────────────────────
# --------------------------------------------------

t_stat, p_val = stats.ttest_ind(pe_eyes_open, pe_eyes_closed, equal_var=False)
print(f"t‑test EO vs EC → t = {t_stat:.2f},  p = {p_val:.4e}")

# Tiny helper to turn p‑value into stars (used below if you want to annotate)

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

# --------------------------------------------------
# 6) VISUALISATION  ───────────────────────────────
# --------------------------------------------------

plt.figure(figsize=(8, 6))
box = plt.boxplot(
    [pe_eyes_closed, pe_eyes_open],
    patch_artist=True,
    boxprops=dict(facecolor="#FFB6C1", color="black"),  # pink fill, black edges
    medianprops=dict(color="black")
)

plt.xticks([1, 2], ["EC (Eyes Closed)", "EO (Eyes Open)"])
plt.ylabel("Permutation Entropy")
plt.title("Permutation Entropy: EO vs EC")

# Optional: annotate p‑value
plt.text(1.5, max(np.max(pe_eyes_closed), np.max(pe_eyes_open)) * 1.05,
         get_p_asterisks(p_val), ha="center", va="bottom", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()
