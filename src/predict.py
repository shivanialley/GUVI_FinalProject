import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.config import IMG_SIZE, CLASS_NAMES, MODEL_PATH

model = load_model(MODEL_PATH)

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    return CLASS_NAMES[np.argmax(pred)]
