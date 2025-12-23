import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from src.config import CLASS_NAMES

st.set_page_config("COVID X-ray Detection")

model = load_model("models/covid_model.h5")

st.title("🦠 COVID-19 X-ray Classification")

uploaded = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).resize((224,224))
    st.image(image, caption="Uploaded X-ray")

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    result = CLASS_NAMES[np.argmax(pred)]

    st.success(f"Prediction: **{result}**")
