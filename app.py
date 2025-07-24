import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# App UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to check if it has a tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((150,150))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
    st.subheader("Result: " + result)
