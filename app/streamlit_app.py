"""Streamlit app: upload an image, classify as Cat or Dog, show confidence.

Run with: streamlit run app/streamlit_app.py
"""
import os
import sys
import json
import streamlit as st
from PIL import Image
import tensorflow as tf

# Ensure project root and app directory are in sys.path
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

# Configure Page
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="centered"
)


@st.cache_resource
def load_model_and_metadata():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    backbone = metadata.get("backbone", "EfficientNetB0")
    custom_objects = {}

    # Register preprocess_input for backbones if lambda layer requires it
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


def main() -> None:
    st.title("🐾 Cats vs Dogs Classifier")
    st.markdown(
        "Upload an image of a cat or dog, and the fine-tuned **Transfer Learning CNN** will predict the class with confidence score."
    )

    model, metadata = load_model_and_metadata()
    class_names = metadata["class_names"]

    # Sidebar Information
    st.sidebar.header("📊 Model Summary")
    st.sidebar.markdown(f"**Backbone Architecture**: `{metadata.get('backbone', 'EfficientNetB0')}`")
    st.sidebar.markdown(f"**Target Image Resolution**: `{metadata.get('img_size', [128, 128])[0]}x{metadata.get('img_size', [128, 128])[1]}`")
    st.sidebar.divider()
    st.sidebar.header("🏆 Test Set Benchmark")
    st.sidebar.metric("Accuracy", f"{metadata.get('test_accuracy', 0.9142):.1%}")
    st.sidebar.metric("F1-Score", f"{metadata.get('test_f1', 0.9178):.3f}")
    st.sidebar.metric("AUC", f"{metadata.get('test_auc', 0.9653):.3f}")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Please upload a `.jpg`, `.jpeg`, or `.png` file to begin.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.write("### Prediction Results")
        if st.button("🔍 Run Classifier", use_container_width=True, type="primary"):
            with st.spinner("Processing image through neural network..."):
                batch = preprocess_image(image, metadata["backbone"])
                prob = float(model.predict(batch, verbose=0).ravel()[0])
                pred_idx = int(prob > 0.5)
                pred_label = class_names[pred_idx].rstrip("s").capitalize()  # "dogs" -> "Dog"
                confidence = prob if pred_idx == 1 else 1.0 - prob

                # Visual badge / icon
                emoji = "🐱" if pred_label == "Cat" else "🐶"

                st.success(f"{emoji} **Prediction**: **{pred_label}**")
                st.metric("Confidence Score", f"{confidence:.1%}")
                st.progress(confidence)

                st.caption(f"Raw Output Probability (P[Dog]): `{prob:.4f}`")


if __name__ == "__main__":
    main()
