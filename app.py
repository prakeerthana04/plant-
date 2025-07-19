# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="🌿 Plant Health Predictor", layout="centered")
st.title("🌱 Plant Health Predictor 🌼")
st.markdown("📸 Upload a leaf image, and let the AI tell you if it's healthy or diseased!")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")  # ensure this matches your filename
    return model

model = load_model()

uploaded = st.file_uploader("📤 Upload leaf image:", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="🌿 Uploaded Image", use_column_width=True)
    st.write("🔍 Analyzing...")
    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255.0, 0)
    preds = model.predict(arr)[0]
    classes = ["Healthy ✅", "Diseased ❌"]
    result = classes[np.argmax(preds)]
    st.success(f"**Prediction:** {result}")
    st.balloons()
