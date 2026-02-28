# Neural Circuit Electrophysiology Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active_Development-orange)

## Overview

This repository contains a comprehensive suite of Python scripts designed for the analysis of in vivo electrophysiological data. The pipeline is specifically tailored for processing single-unit activity (SUA) and local field potentials (LFP) recorded from neural circuits (e.g., mPFC).

The toolkit covers the entire analytical workflow, from basic neuronal property characterization (firing rate, waveform, ISI) to advanced functional connectivity analysis using strict statistical thresholds.

## Key Features

### 1. Basic Neuronal Characterization

- **Waveform Analysis:** Quantification of Peak-Valley duration, FWHM, and amplitude to classify neurons (e.g., Pyramidal vs. Interneurons).
- **Firing Properties:** Calculation of average firing rates, Inter-Spike Interval (ISI) distributions, and coefficient of variation ($CV_{ISI}$).
- **Burst Metrics:** Detection of burst events, burst rates, and intra-burst frequencies.

### 2. Synaptic Connectivity Analysis (CCH)

This pipeline implements two distinct, rigorously validated algorithms to detect functional monosynaptic connections:

* **Gaussian Convolution Method (Strict):** Based on **Sauer & Bartos (2022)**. Uses a hollow Gaussian kernel to estimate the baseline and strictly detects sharp peaks/troughs in the monosynaptic window (0.8–2.8 ms).
* **Jittering Resampling Method:** Based on **Fujisawa et al. (2008)**. Generates 1,000 surrogate datasets by jittering spike times (-5ms to +5ms) to establish pointwise significance bands (99.5% for excitation, 0.5% for inhibition).

Supported connection types:

- **Excitatory:** PYR $\to$ INT, PYR $\to$ PYR
- **Inhibitory:** INT $\to$ PYR, INT $\to$ INT

### 3. Population Dynamics & LFP

- **Cell Assembly Detection:** Identification of co-active neuronal ensembles using ICA (Independent Component Analysis).
- **LFP Analysis:** Power Spectral Density (PSD) calculation (Welch's method) across Delta, Theta, Alpha, Beta, and Gamma bands.
- **LFP Spectrograms:** Time-frequency analysis using **Complex Morlet Wavelet Transform (CWT)** to visualize dynamic power changes (1-100 Hz). Includes raw LFP waveform extraction and visualization.
- **K-means Clustering:** Unsupervised classification of neuron types based on waveform features.

## Requirements

* Python 3.8 or higher
* Operating System: Windows, macOS, or Linux (GUI features rely on `tkinter`)

## Installation

1. Clone this repository:

   ```bash
   git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
   cd Your-Repo-Name
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## File Structure & Usage Guide

The scripts are numbered to suggest a logical workflow, though they can be run independently. All scripts feature a **Graphical User Interface (GUI)** for file selection.

### Part I: Basic Characterization

| Script Name                          | Function                                                     |
| :----------------------------------- | :----------------------------------------------------------- |
| `1_Frequency_Calculator.py`          | Calculates mean firing rates for all neurons in a session.   |
| `2_Waveform_Calculator.py`           | Extracts waveform features (FWHM, Peak-Valley time) for neuron classification. |
| `3_Autocorrelogram.py`               | Generates autocorrelograms (ACG) to assess rhythmicity and refractory periods. |
| `3_Autocorrelogram_Normalization.py` | Normalizes ACGs for population averaging.                    |
| `4_ISI_Calculator_and_Histogram.py`  | Computes ISI distributions and plots histograms (log-scale). |
| `5_CVisi_Calculator.py`              | Calculates the Coefficient of Variation of ISI to assess firing irregularity. |
| `6_K-means_Analysis.py`              | Performs K-means clustering to separate putative PYR and INT units. |
| `7_Burst_Metrics.py`                 | Quantifies bursting activity (burst rate, spikes per burst, etc.). |

### Part II: Synaptic Connectivity (Cross-Correlation)

| Script Name                     | Method    | Direction     | Type       |
| :------------------------------ | :-------- | :------------ | :--------- |
| `8_CCH_Gaussian...PYR-INT.py`   | Gaussian  | PYR $\to$ INT | Excitatory |
| `9_CCH_Gaussian...INT-PYR.py`   | Gaussian  | INT $\to$ PYR | Inhibitory |
| `10_CCH_Gaussian...PYR-PYR.py`  | Gaussian  | PYR $\to$ PYR | Excitatory |
| `11_CCH_Gaussian...INT-INT.py`  | Gaussian  | INT $\to$ INT | Inhibitory |
| `12_CCH_Jittering...PYR-INT.py` | Jittering | PYR $\to$ INT | Excitatory |
| `13_CCH_Jittering...INT-PYR.py` | Jittering | INT $\to$ PYR | Inhibitory |
| `14_CCH_Jittering...PYR-PYR.py` | Jittering | PYR $\to$ PYR | Excitatory |
| `15_CCH_Jittering...INT-INT.py` | Jittering | INT $\to$ INT | Inhibitory |

### Part III: Network Dynamics

| Script Name               | Function                                                     |
| :------------------------ | :----------------------------------------------------------- |
| `16_Assembly_Analysis.py` | Detects cell assemblies and co-firing patterns using ICA.    |
| `17_LFP_Analysis.py`      | Batch processing of LFP data: Notch filtering, Bandpass filtering, and PSD. |
| `17_LFP_Spectrogram.py`   | **(New)** Time-Frequency Analysis (CWT) and Raw LFP Waveform plotting. |

### Part IV: Downsampling Analysis

| Script Name               | Function                                                     |
| :------------------------ | :----------------------------------------------------------- |
| `18_CCH_all in one_resampling.py` | Includes rigorous unidirectional downsampling engines specifically tailored for handling experimental vs. control group disparities (e.g., CUMS vs. Control), preventing baseline firing rate biases from skewing connectivity results.    |
| `19_Assembly analysis_resampling.py`      | Advanced ICA cell assembly detection with specific population down-sampling control (Ctrl/CUMS) and pairwise sync validation. |

## Input Data Format

* **Spike Data:** Standard `.xlsx` files where each column represents a neuron and rows represent spike timestamps (in seconds).
* **LFP Data:** `.xlsx` files containing `Timestamp` and `Voltage` columns.
* **Neuron Info:** For connectivity scripts, the first two rows of the input Excel file must specify:
  * Row 1: Neuron ID
  * Row 2: Neuron Type (1 = Pyramidal, 0 = Interneuron)

## References

If you use the connectivity algorithms in this toolbox, please cite the respective methodologies:

1.  **Gaussian Convolution:** Sauer, J. F., & Bartos, M. (2022). *Disrupted hippocampal-prefrontal synchrony in a mouse model of schizophrenia*. eLife, 11, e78428.
2.  **Jittering Resampling:** Fujisawa, S., Amarasingham, A., Harrison, M. T., & Buzsáki, G. (2008). *Behavior-dependent short-term assembly dynamics in the medial prefrontal cortex*. Nature Neuroscience, 11(7), 823–833.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
