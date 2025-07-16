import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# Must be first Streamlit command
st.set_page_config(page_title="Animal Image Classifier", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("animal_model.h5")
    return model

model = load_model()

# Class labels
class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant',
               'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

# Streamlit UI
st.markdown("""
    <style>
        .main-title {
            font-size: 2.8rem;
            color: #33ccff;
            font-weight: bold;
            text-align: center;
            padding-top: 15px;
            animation: fadeInDown 1s ease-in-out;
        }
        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #aaa;
            padding-top: 40px;
        }
        .fade-in {
            animation: fadeIn 1.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🐾 Animal Image Classifier</div>', unsafe_allow_html=True)

st.markdown("""
<div class='fade-in'>
<p style='text-align: center; font-size: 1.1rem;'>
Upload an image of an animal below and our intelligent model will classify which animal it is.
Trained on 15 different animal categories using transfer learning with MobileNetV2.
</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an animal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"**Predicted Animal:** {predicted_class} ({confidence:.2f}% confidence)")

st.markdown("""
<div class="footer">
    Animal Image Classification App using Transfer Learning | Created by Lakshay Rathee<br>
    Powered by TensorFlow & Streamlit | Dataset: 15 Animal Classes
</div>
""", unsafe_allow_html=True)