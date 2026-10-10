# Particle Flow Network (PFN) for Top Quark Tagging

## Overview
This repository implements an end-to-end Deep Learning pipeline for Top Quark Tagging using CERN ATLAS Open Data. The model leverages a Particle Flow Network (PFN) based on the Deep Sets architecture to effectively classify Top Quark jets from background QCD jets by processing variable-length, permutation-invariant particle inputs.

## Pipeline Architecture
1. **High-Speed Data Fetching:** Utilizes CERN's XRootD client (`xrdcp`) to bypass HTTP limitations and download large `.h5` datasets at maximum Grid speeds.
2. **Physical Data Preprocessing (`ColabTopTaggingDataset`):**
   - **Log-Scaling:** Applies `np.log1p` to $p_T$ and Energy values to compress high-variance distributions while strictly preserving zero-padding structures.
   - **Geometric Centering:** Shifts $\eta$ and $\phi$ spatial coordinates relative to the leading particle (highest $p_T$) within each jet to establish rotational and spatial invariance.
3. **PFN / Deep Sets Model (`ParticleFlowNetwork`):**
   - **$\Phi$ (Phi) Network:** Applies 1D Convolutions (`nn.Conv1d`) independently to each particle to extract localized feature representations.
   - **Sum Pooling:** Utilizes a permutation-invariant aggregation operation (`torch.sum`) across the particle dimension to construct a unified global jet representation.
   - **$F$ Network:** Passes the aggregated jet representation through fully connected layers (`nn.Linear`) to output a single binary classification logit.
4. **Dynamic Training Phase:** The network is trained over 5 Epochs using the Adam optimizer alongside a Weighted Binary Cross Entropy Loss (`BCEWithLogitsLoss`). A `StepLR` scheduler is employed to automatically decay the learning rate by 50% every 2 epochs, ensuring stable convergence.
5. **Inference & Evaluation:**
   - Computes Receiver Operating Characteristic (ROC) curves, achieving an outstanding **Test AUC of 0.9444**.
   - Mathematically isolates the optimal decision boundary by maximizing the J-statistic, yielding an **Optimal Threshold of 0.5150**.
   - Provides full classification reports and visualizes output probability densities (Top Signal vs. Background) via Matplotlib histograms.
