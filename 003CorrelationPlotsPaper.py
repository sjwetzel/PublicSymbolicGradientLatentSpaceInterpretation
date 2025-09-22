# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 14:18:39 2025

@author: sebas
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Settings
# ---------------------------
PLOTS_PER_PAGE = 12
N_ROWS, N_COLS = 4, 3   # 4 rows × 3 columns
FIGSIZE = (14, 18)
POINT_SIZE = 14
ALPHA = 0.65
POINT_COLOR = "steelblue"   # same color for all plots

LABEL_FONTSIZE = 18   # increased axis label size
TICK_FONTSIZE = 16    # increased tick label size
TITLE_FONTSIZE = 18
SUPTITLE_FONTSIZE = 20

# ---------------------------
# Collect folders
# ---------------------------
parent_dir = os.getcwd()
folders = sorted([f for f in os.listdir(parent_dir) if os.path.isdir(f)])

# ---------------------------
# Prepare figure
# ---------------------------
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=FIGSIZE, constrained_layout=True)
axes = axes.flatten()

# ---------------------------
# Plot data
# ---------------------------
for i, folder in enumerate(folders[:PLOTS_PER_PAGE]):
    latent_data_path = os.path.join(folder, "latent_data")

    try:
        invariant = np.load(os.path.join(latent_data_path, "invariant.npy"))
        latent = np.load(os.path.join(latent_data_path, "latent.npy"))
    except FileNotFoundError:
        print(f"Skipping {folder}: missing latent_data files")
        continue

    axes[i].scatter(
        invariant, latent,
        c=POINT_COLOR,
        s=POINT_SIZE,
        alpha=ALPHA,
        edgecolors="none"
    )

    # Titles and labels
    axes[i].set_title(folder, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=8)
    axes[i].set_xlabel("Invariant", fontsize=LABEL_FONTSIZE)
    axes[i].set_ylabel("Latent", fontsize=LABEL_FONTSIZE)

    # Style tweaks
    axes[i].grid(True, linestyle="--", alpha=0.3)
    axes[i].tick_params(axis="both", labelsize=TICK_FONTSIZE)

# ---------------------------
# Hide unused axes
# ---------------------------
for j in range(len(folders), PLOTS_PER_PAGE):
    fig.delaxes(axes[j])

# ---------------------------
# Final display
# ---------------------------
fig.suptitle("Latent Space Correlation Plots", fontsize=SUPTITLE_FONTSIZE, fontweight="bold", y=1.02)
plt.show()
