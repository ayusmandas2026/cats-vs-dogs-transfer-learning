
"""Preprocessing shared by training and the Streamlit app.

Keeping this logic in one module and importing it from both places is what
guarantees train/serve parity (Section 12 of the notebook).
"""
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (128, 128)


def preprocess_image(pil_image: Image.Image, backbone_name: str) -> np.ndarray:
    """Convert a PIL image into a (1, 128, 128, 3) float array ready for
    model.predict(), matching the exact steps used during training:
    force RGB -> resize -> add batch dim. Backbone-specific normalization
    (preprocess_input) is applied inside the saved model itself, so it is
    NOT duplicated here.
    """
    image = pil_image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)
