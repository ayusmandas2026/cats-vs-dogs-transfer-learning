# 🐾 Cats vs. Dogs — Image Classification Using CNNs with Transfer Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Learning and Computer Vision repository developed for **Lab Assignment 03 ("Cats Versus Dogs – Image Classification Using CNNs with Transfer Learning")**. 

This project evaluates **five pre-trained ImageNet architectures** (`EfficientNetB0`, `MobileNetV2`, `VGG16`, `ResNet50`, and `Xception`) across feature extraction and fine-tuning stages, performs hyperparameter search, conducts failure-mode error analysis, and deploys a multi-tab production Streamlit web application.

---

## 📌 Project Architecture

```text
cats-vs-dogs-transfer-learning/
├── MLPWP_LAB_3.ipynb         # Master Jupyter notebook containing complete end-to-end pipeline
├── app/                      # Streamlit production deployment modules
│   ├── preprocessing.py      # Shared image preprocessing (ensures train-serve parity)
│   └── streamlit_app.py      # Multi-tab interactive Streamlit web dashboard
├── Dataset/                  # Structured binary classification dataset
│   ├── train/                # Training set subdirectories: cats/, dogs/
│   └── test/                 # Held-out evaluation set: cats/, dogs/
├── models/                   # Serialized model weights & production metadata
│   └── final/
│       ├── best_model.keras  # Production model weights (Fine-tuned EfficientNetB0)
│       └── model_metadata.json # Production metadata & evaluation benchmarks
├── reports/                  # Generated diagnostic visualizations & CSV benchmarks
│   ├── figures/              # Confusion matrix, ROC curve, training curves, error plots
│   └── metrics/              # CSV summaries (model_comparison.csv, hyperparameter_tuning.csv)
├── requirements.txt          # Python dependencies for local and cloud deployment
├── .gitignore                # Version control exclusion configuration
└── README.md                 # Project documentation
```

---

## 🏆 Model Comparison & Experimental Results

All architectures were evaluated on the **held-out test set** across predictive accuracy, parameter count, training time, and inference latency:

| Model Architecture | Total Parameters | Trainable Params | Test Accuracy | Precision | Recall | F1-Score | ROC AUC | Training Time | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **EfficientNetB0** | **4,213,668** | **1,064,497** | **92.14%** | **0.8831** | **0.9714** | **0.9252** | **0.9651** | **237.8s** | **134.1ms** |
| 🥈 **MobileNetV2** | 2,422,081 | 1,358,977 | 89.29% | 0.8313 | 0.9857 | 0.9020 | 0.9759 | 146.4s | 122.4ms |
| 🥉 **VGG16** | 14,780,481 | 7,145,217 | 88.57% | 0.8462 | 0.9429 | 0.8919 | 0.9696 | 648.8s | 216.6ms |
| **ResNet50** | 23,850,113 | 4,721,921 | 87.14% | 0.8250 | 0.9429 | 0.8800 | 0.9710 | 284.2s | 376.4ms |
| **Xception** | 21,123,881 | 5,749,505 | 87.14% | 0.8421 | 0.9143 | 0.8767 | 0.9582 | 297.4s | 147.6ms |

### 💡 Champion Model Selection Rationale
**`EfficientNetB0`** was selected as the final production model because it delivers the **highest test accuracy (92.14%) and F1-score (0.9252)** while requiring only **~4.21 million parameters** (~82% smaller than ResNet50), balancing high predictive accuracy with edge deployment efficiency.

---

## ⚡ Quick Start & Local Execution

### 1. Clone the Repository
```bash
git clone https://github.com/ayusmandas2026/cats-vs-dogs-transfer-learning.git
cd cats-vs-dogs-transfer-learning
```

### 2. Set Up Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501` to use the multi-tab interactive classifier.

---

## 📊 Streamlit App Features

- **🔮 Interactive Classifier**: Test images by uploading custom `.jpg`/`.png` files OR selecting pre-loaded test samples with instant confidence meters.
- **📊 Benchmarks & Leaderboard**: Interactive comparison tables and bar charts across parameter sizes, latencies, and metrics.
- **📈 Diagnostic Curves**: View confusion matrices, ROC curves, and multi-stage training histories.
- **🔍 Qualitative Error Analysis**: Inspect categorized failure mode samples (False Positives, False Negatives, Low Confidence instances).
- **📜 Executive Summary**: Complete Lab Assignment compliance overview and model selection rationale.

---

## 🔬 Key Methodology Highlights

1. **Two-Stage Transfer Learning**:
   - *Stage 1 (Feature Extraction)*: Convolutional backbones frozen (`layer.trainable = False`), training only Dense head.
   - *Stage 2 (Fine-Tuning)*: Unfreezing top $N$ layers with a low learning rate ($10^{-5}$). `BatchNormalization` layers remain frozen to preserve running mean/variance statistics.
2. **Train/Serve Parity**:
   - `app/preprocessing.py` standardizes input images ($128 \times 128 \times 3$, RGB conversion, batch expansion) identically across training notebooks and the live Streamlit server.

---

## 📝 License
Distributed under the **MIT License**. See `LICENSE` for details.
