# src/config.py

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 3

CLASS_NAMES = ["COVID", "Viral_Pneumonia", "Normal"]

# ✅ FIXED PATHS (according to your folder)
TRAIN_DIR = "data/Covid19-dataset/train"
TEST_DIR  = "data/Covid19-dataset/test"

MODEL_PATH = "models/covid_model.h5"