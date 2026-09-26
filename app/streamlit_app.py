"""Streamlit Application: Cats vs Dogs Image Classification

Lab Assignment 03 - Software Development Project Lab
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from preprocessing import preprocess_image
except ImportError:
    from app.preprocessing import preprocess_image

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final", "best_model.keras")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "final", "model_metadata.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")

# Page Configuration
st.set_page_config(
    page_title="Cats vs Dogs Classification — SDP Lab 3",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling - Minimalist Academic Theme
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
    }
    .prediction-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #0F172A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_metadata():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    backbone = metadata.get("backbone", "EfficientNetB0")
    custom_objects = {}

    backbone_map = {
        "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
        "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "VGG16": tf.keras.applications.vgg16.preprocess_input,
        "ResNet50": tf.keras.applications.resnet50.preprocess_input,
        "Xception": tf.keras.applications.xception.preprocess_input,
    }
    if backbone in backbone_map:
        custom_objects["preprocess_input"] = backbone_map[backbone]

    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    except Exception:
        model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

    return model, metadata


@st.cache_data
def load_metrics_tables():
    comp_path = os.path.join(METRICS_DIR, "model_comparison.csv")
    tune_path = os.path.join(METRICS_DIR, "hyperparameter_tuning.csv")

    df_comp = pd.read_csv(comp_path) if os.path.exists(comp_path) else None
    df_tune = pd.read_csv(tune_path) if os.path.exists(tune_path) else None

    return df_comp, df_tune


def main():
    model, metadata = load_model_and_metadata()
    df_comp, df_tune = load_metrics_tables()

    # Sidebar Summary
    st.sidebar.title("SDP Lab Assignment 3")
    st.sidebar.markdown("**Topic**: CNN Transfer Learning for Image Classification")
    st.sidebar.divider()

    st.sidebar.markdown("### Selected Model Configuration")
    st.sidebar.write(f"**Backbone**: {metadata.get('backbone', 'EfficientNetB0')}")
    st.sidebar.write(f"**Input Size**: {metadata.get('img_size', [128, 128])[0]} × {metadata.get('img_size', [128, 128])[1]} × 3")
    st.sidebar.write(f"**Total Parameters**: ~4.21 Million")
    st.sidebar.divider()

    st.sidebar.markdown("### Test Performance")
    st.sidebar.write(f"**Accuracy**: {metadata.get('test_accuracy', 0.9143):.2%}")
    st.sidebar.write(f"**F1 Score**: {metadata.get('test_f1', 0.9178):.4f}")
    st.sidebar.write(f"**ROC AUC**: {metadata.get('test_auc', 0.9653):.4f}")

    # Header
    st.markdown('<div class="main-title">Cats vs. Dogs Image Classification Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Comparative Analysis of Transfer Learning Architectures & Single-Image Inference System</div>',
        unsafe_allow_html=True,
    )

    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Image Inference",
        "Model Comparison",
        "Training & Evaluation Plots",
        "Error Analysis",
        "Lab Report Summary",
    ])

    # ---------------------------------------------------------
    # TAB 1: Image Inference
    # ---------------------------------------------------------
    with tab1:
        st.markdown("### Single-Image Inference")
        st.write("Upload an image file or choose a sample image from the dataset to test the classifier.")

        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("#### Input Selection")
            mode = st.radio("Input Method", ["Upload Image", "Select Pre-loaded Sample Image"], horizontal=True)

            img = None
            if mode == "Upload Image":
                uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    img = Image.open(uploaded_file)
            else:
                sample_map = {
                    "Cat Test Image 1": os.path.join(PROJECT_ROOT, "Dataset", "test", "cats", "cat_1.jpg"),
                    "Cat Test Image 2": os.path.join(PROJECT_ROOT, "Dataset", "test", "cats", "cat_106.jpg"),
                    "Dog Test Image 1": os.path.join(PROJECT_ROOT, "Dataset", "test", "dogs", "dog_114.jpg"),
                    "Dog Test Image 2": os.path.join(PROJECT_ROOT, "Dataset", "test", "dogs", "dog_123.jpg"),
                }
                sample_choice = st.selectbox("Select Sample", list(sample_map.keys()))
                path = sample_map[sample_choice]
                if os.path.exists(path):
                    img = Image.open(path)

            if img is not None:
                st.image(img, caption="Input Image Preview", use_container_width=True)

        with col_right:
            st.markdown("#### Inference Results")
            if img is None:
                st.info("Select or upload an image on the left to display prediction results.")
            else:
                if st.button("Run Model Inference", type="primary", use_container_width=True):
                    batch = preprocess_image(img, metadata["backbone"])
                    prob = float(model.predict(batch, verbose=0).ravel()[0])
                    pred_idx = int(prob > 0.5)
                    class_names = metadata["class_names"]
                    pred_label = class_names[pred_idx].rstrip("s").capitalize()  # "dogs" -> "Dog"
                    confidence = prob if pred_idx == 1 else 1.0 - prob

                    st.markdown("---")
                    st.markdown(f'<div class="prediction-header">Prediction: <b>{pred_label}</b></div>', unsafe_allow_html=True)
                    st.write(f"**Confidence Score**: `{confidence:.2%}`")
                    st.progress(confidence)

                    st.markdown("##### Class Probability Distribution")
                    cat_p = 1.0 - prob
                    dog_p = prob
                    prob_df = pd.DataFrame({"Probability": [cat_p, dog_p]}, index=["Cat", "Dog"])
                    st.bar_chart(prob_df, height=180)

                    with st.expander("Technical Execution Details"):
                        st.json({
                            "Selected Backbone": metadata.get("backbone"),
                            "Preprocessed Tensor Shape": list(batch.shape),
                            "Raw Output Probability P(Dog)": float(round(prob, 6)),
                            "Decision Threshold": 0.5,
                            "Predicted Index": pred_idx,
                            "Train/Serve Parity": "Enforced via shared preprocessing module",
                        })

    # ---------------------------------------------------------
    # TAB 2: Model Comparison
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### Transfer Learning Backbones Benchmark")
        st.write("Controlled evaluation of five pre-trained CNN backbones on the held-out test dataset.")

        if df_comp is not None:
            st.markdown("#### Quantitative Performance Table")
            st.dataframe(
                df_comp.style.format({
                    "Test Accuracy": "{:.2%}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                    "AUC": "{:.4f}",
                    "Parameters": "{:,}",
                    "Trainable Parameters": "{:,}",
                    "Training Time (s)": "{:.1f}",
                    "Inference (ms/img)": "{:.1f}",
                }),
                use_container_width=True,
            )

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("#### Test Accuracy by Architecture")
                st.bar_chart(df_comp.set_index("Model")[["Test Accuracy"]])
            with col_c2:
                st.markdown("#### Inference Latency (ms / image)")
                st.bar_chart(df_comp.set_index("Model")[["Inference (ms/img)"]])
        else:
            st.warning("Model comparison data not found.")

        st.markdown("---")
        st.markdown("### Hyperparameter Tuning Results")
        st.write("Validation results for top finalist architectures across varying learning rates, dropout, and unfrozen depths.")
        if df_tune is not None:
            st.dataframe(
                df_tune.style.format({
                    "Best Val Accuracy": "{:.2%}",
                    "LR": "{:.1e}",
                }),
                use_container_width=True,
            )

    # ---------------------------------------------------------
    # TAB 3: Training & Evaluation Plots
    # ---------------------------------------------------------
    with tab3:
        st.markdown("### Performance Plots and Training History")
        st.write("Diagnostic figures generated during dataset exploration, model training, and final test set evaluation.")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Confusion Matrix")
            p_cm = os.path.join(FIGURES_DIR, "confusion_matrix.png")
            if os.path.exists(p_cm):
                st.image(p_cm, caption="Confusion Matrix on Test Dataset", use_container_width=True)

            st.markdown("#### Training and Validation Loss/Accuracy")
            p_tc = os.path.join(FIGURES_DIR, "training_curves.png")
            if os.path.exists(p_tc):
                st.image(p_tc, caption="Stage 1 (Feature Extraction) and Stage 2 (Fine-Tuning) Curves", use_container_width=True)

        with c2:
            st.markdown("#### ROC Curve")
            p_roc = os.path.join(FIGURES_DIR, "roc_curve.png")
            if os.path.exists(p_roc):
                st.image(p_roc, caption="ROC Curve (Test AUC = 0.965)", use_container_width=True)

            st.markdown("#### Data Augmentation Pipeline")
            p_aug = os.path.join(FIGURES_DIR, "augmentation_examples.png")
            if os.path.exists(p_aug):
                st.image(p_aug, caption="Sample Image Transformations (Rotation, Zoom, Flip, Translation)", use_container_width=True)

    # ---------------------------------------------------------
    # TAB 4: Error Analysis
    # ---------------------------------------------------------
    with tab4:
        st.markdown("### Error Analysis and Failure Mode Inspection")
        st.write("Qualitative review of correctly classified images, misclassifications, and low-confidence edge cases.")

        e1, e2 = st.columns(2)

        with e1:
            st.markdown("#### Correct Classification Examples")
            p_ec = os.path.join(FIGURES_DIR, "error_correct.png")
            if os.path.exists(p_ec):
                st.image(p_ec, caption="Sample Correct Predictions", use_container_width=True)

            st.markdown("#### False Positives (Cat misclassified as Dog)")
            p_fp = os.path.join(FIGURES_DIR, "error_false_positive_cat_to_dog.png")
            if os.path.exists(p_fp):
                st.image(p_fp, caption="Cat Images Predicted as Dog", use_container_width=True)

        with e2:
            st.markdown("#### False Negatives (Dog misclassified as Cat)")
            p_fn = os.path.join(FIGURES_DIR, "error_false_negative_dog_to_cat.png")
            if os.path.exists(p_fn):
                st.image(p_fn, caption="Dog Images Predicted as Cat", use_container_width=True)

            st.markdown("#### Low-Confidence Predictions (0.40 – 0.60 Probability Band)")
            p_lc = os.path.join(FIGURES_DIR, "error_low_confidence.png")
            if os.path.exists(p_lc):
                st.image(p_lc, caption="Ambiguous or Borderline Test Cases", use_container_width=True)

        st.markdown("---")
        st.markdown("#### Observed Failure Causes")
        st.markdown(
            """
            - **Partial Visibility & Cropping**: Images where key facial features (ears, muzzle) are cropped tend to reduce prediction certainty.
            - **Background Interference**: Complex or cluttered backgrounds can draw spatial attention away from primary subjects.
            - **Lighting and Pose Variation**: Extreme lighting contrast or non-standard sleeping postures deviate from standard training samples.
            """
        )

    # ---------------------------------------------------------
    # TAB 5: Lab Report Summary
    # ---------------------------------------------------------
    with tab5:
        st.markdown("### Lab Assignment 03 - Summary & Conclusions")
        st.markdown(
            """
            #### 1. Problem Statement & Objectives
            This project investigates transfer learning techniques for binary image classification of cats and dogs using Convolutional Neural Networks (CNNs). The primary objective is to evaluate multiple pre-trained ImageNet backbones to identify the model offering the best trade-off between predictive accuracy, model size, and inference latency.

            #### 2. Summary of Methodology
            - **Preprocessing & Augmentation**: Images resized to 128 × 128 × 3; pixel normalization applied per backbone specification. Training data augmented with random rotations, flips, zoom, and translation.
            - **Stage 1 (Feature Extraction)**: Pre-trained convolutional base frozen; trained custom classification head (GlobalAveragePooling2D + Dense).
            - **Stage 2 (Fine-Tuning)**: Unfroze top layers with reduced learning rate ($10^{-5}$); kept BatchNormalization layers frozen to protect running statistics.

            #### 3. Model Selection Rationale
            **EfficientNetB0** was selected as the final production model based on empirical evidence:
            - **Predictive Performance**: Highest test accuracy (92.14%), F1 score (0.9252), and AUC (0.9651).
            - **Efficiency**: Requires ~4.21M parameters, making it significantly smaller than ResNet50 (23.8M) and VGG16 (14.8M).
            - **Inference Speed**: Fast response time of ~134 ms per image.
            """
        )


if __name__ == "__main__":
    main()
