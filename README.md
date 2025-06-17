# Final Degree Project – EEG Signal Analysis

This repository contains all the scripts and resources used for my final degree project in Bioinformatics. The goal of the project was to analyze EEG signals in order to compare brain states — Eyes Open (EO) vs. Eyes Closed (EC) — using various entropy-based and statistical features.

---

## What this project is about

I worked on analyzing EEG recordings using different metrics like **Permutation Entropy (PE)**, **Spatial Permutation Entropy (SPE)**, and classical statistical features such as **Mean**, **Variance**, **Skewness**, **Kurtosis**, etc. I also explored how preprocessing, time duration, and electrode configurations affect the results. The aim was to understand how these brain states differ and which features help distinguish them best.

---

## Repository structure

### `SPE_dificult_conditions/`

Scripts for analyzing **Spatial Permutation Entropy (SPE)** under different configurations and time durations:

- `31_17_17P.py`: Compares EO vs. EC using 31 and 17 electrodes, showing boxplots of horizontal and vertical entropy.
- `boxplot_spatial.py`: Visualizes vertical boxplots comparing EC and EO with and without artifacts.
- `p_value_spatial.py`: Computes SPE using parallel processing and tests for significance (prints p-values).
- `tiempo.py`: Analyzes how SPE evolves over time in EO vs. EC using artifact-free signals.
- `boxplot_p_val_generardato.py`: Computes PE for EC/EO using multiprocessing and plots a boxplot with significance annotation.

### `representaciones_topomaps_boxplots/`

Scripts for generating **topographic maps** and **boxplots** using EEG features:

- `boxplot_represent_pval.py`: Boxplots comparing EC vs. EO for PE, skewness and kurtosis with p-values.
- `comb_boxplots.py`: Compares metrics with and without artifacts using side-by-side boxplots.
- `comb_topomap.py`: Generates topographic maps (EC, EO, difference, p-value) for each EEG feature.
- `temporal_boxplot_64_31_17.py`: Shows PE differences using 64, 31, and 17 electrodes over time.

### `scripts_different_metrics/`

Scripts that compute classical **statistical features** for EEG signals:

- `egg_analysis_mean.py`: Mean signal per channel.
- `egg_analysis_var.py`: Signal variance.
- `egg_analysis_kurtosis.py`: Kurtosis per channel.
- `egg_analysis_IQR.py`: Interquartile Range.
- `egg_analysis_MAD.py`: Median Absolute Deviation.
- `egg_analysis_2_with_skew.py`: Combines PE and skewness in one figure.

---

##  Other important files (main folder)

- `egg_analysis_2_with_pval.py`: Computes PE and performs a t-test between EO and EC, then plots the results.
- `egg_utils_2.py`: Core EEG class used in all scripts to load, preprocess and analyze EEG signals.
- `ICA_Corrected.py`: Removes eye blink artifacts using ICA from the MNE library.
- `PSD.py`: Script for computing Power Spectral Density from EEG data.
- `plots_whole_time_serie.py`: Evaluates how PE changes over different time windows for EO and EC.
- `make_figs.m`: MATLAB script used to organize or clean plots for the final report.
- `spe_summary_by_subject.csv`: Summary table of SPE values per subject.

---

## Data info

The EEG data comes from **PhysioNet** and consists of `.edf` files for each subject and condition (EO/EC). Some `.npy` files used to store intermediate results are too large to upload here.

File structure used:

- Raw EEG files: `/files-2/`
- Preprocessed matrices: `/vectores/` 


##  Summary of the analysis

- EO vs. EC comparison using PE and SPE.
- Effect of artifacts on entropy-based features.
- Impact of reducing electrode count (from 64 to 31/17).
- How entropy evolves with different signal durations.
- Clear visualizations with statistical testing (boxplots, p-values, topomaps).
