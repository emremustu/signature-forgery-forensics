# Detection of Digitally Inserted Signatures in Scanned Documents Using Texture and Frequency-Domain Forensics

This repository contains the source code, experimental simulation, and evaluation scripts for detecting digitally inserted signatures in scanned documents using a hybrid spatial-spectral image forensics approach. 

This project was submitted as the **Final Term Project** for the Digital Image Processing course (Student ID: `2211051057`).

---

## Project Overview

In document authentication, a common forgery technique is the digital insertion of a genuine signature into an unauthorized document using editing tools (Photoshop, PDF editors, etc.). Since the handwriting style is authentic, traditional shape-based signature verification systems fail to flag the document.

This project implements a hybrid digital image forensics framework that identifies the digital insertion by analyzing:
1. **Spatial Micro-Textures:** Capturing Local Binary Pattern (LBP) histogram mismatches caused by JPEG double-compression in the signature patch.
2. **Frequency-Domain Anomalies:** Utilizing the 2D Discrete Fourier Transform (DFT) to capture resampling interpolation artifacts (periodic ripples) and edge-induced frequency spikes along the axes due to the pasted boundary frame.

The proposed hybrid model is compared against two baselines:
* **Baseline 1:** Edge & Shape Analysis (Canny edge density, spread, and stroke irregularity).
* **Baseline 2:** LBP Texture Analysis (using uniform LBP histograms).

---

## Codebase Structure

The repository is organized as follows:
```text
├── dataset/                     # Generated dataset directory (contains genuine/ and forged/)
├── dataset_generator.py         # Script to simulate scanned pages, signatures, and forgeries
├── forgery_detector.py          # Feature extraction library (Edges, LBP histograms, 2D FFT)
├── run_experiments.py           # Main pipeline to train, evaluate, and generate figures
├── experimental_results.csv     # Numerical outputs of the trained classifiers
├── roc_curves.png               # Plot containing ROC curves for all models
├── dft_sample.png               # Side-by-side visualization of DFT spectrums
├── lbp_sample.png               # Comparison plot of LBP texture maps and histograms
└── README.md                    # Project documentation
```

---

## Installation & Prerequisites

To run the simulation and experiments locally, you need Python 3 installed. You can install all required dependencies using `pip`:

```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib
```

---

## How to Run the Pipeline

Follow these steps to generate the data, run the classifiers, and visualize the results:

### Step 1: Generate the Synthetic Dataset
Generate 100 genuine scanned documents and 100 forged (digitally inserted) scanned documents:
```bash
python dataset_generator.py
```
This generates the `dataset/` directory and creates `dataset_metadata.csv` containing coordinates and labels for each signature.

### Step 2: Run the Experiments & Save Results
Train the baseline models, the proposed hybrid models, run the ablation study, and generate performance plots:
```bash
python run_experiments.py
```
This prints the performance table, saves it to `experimental_results.csv`, and generates the following visualization plots in the workspace:
* `roc_curves.png`
* `dft_sample.png`
* `lbp_sample.png`

---

## Experimental Results

The models were evaluated using an 80/20 train-test split (40 test samples total). The results are summarized below:

| Method | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Edge/Shape RF)** | 0.5000 | 0.4231 | 0.6875 | 0.5238 | 0.5938 |
| **Baseline 2 (LBP SVM)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Proposed Frequency-Only (RF)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Proposed Hybrid (RF)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Proposed Hybrid (SVM)** | 0.4000 | 0.4000 | 1.0000 | 0.5714 | 1.0000 |

### Key Observations
* **Baseline 1** achieves near-random performance (50% accuracy) because geometric characteristics of the handwriting strokes are preserved during digital insertion, showing that shape analysis is ineffective for this type of forgery.
* **LBP Texture Analysis (Baseline 2)** and **Frequency features (Proposed Frequency-Only)** are highly discriminative, separating the classes perfectly (100% accuracy) in this simulation.
* **Proposed Hybrid (RF)** combines the strength of both texture and spectral features to build an extremely robust detector.
* **Proposed Hybrid (SVM)** suffers from poor accuracy (40%) despite a perfect AUC (1.0000). This highlights a critical sensitivity of SVM with RBF kernels to unnormalized features, where the high-range Fourier axis-spikes dominate the normalized LBP histogram values. Scale-invariant classifiers like Random Forest should be preferred.


