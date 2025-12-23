from src.data_loader import load_data
from src.model import build_model
from src.config import EPOCHS, MODEL_PATH

def train_model():
    train_data, test_data = load_data()
    model = build_model()

    model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS
    )

    model.save(MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")
