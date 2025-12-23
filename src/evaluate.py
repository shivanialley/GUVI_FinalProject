import numpy as np
import pandas as pd
import mlflow
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import load_data
from src.config import MODEL_PATH, CLASS_NAMES
from src.logger import logger
import os

def evaluate_model():
    logger.info("Starting evaluation")

    os.makedirs("outputs", exist_ok=True)

    _, test_data = load_data()
    model = load_model(MODEL_PATH)

    y_true = test_data.classes
    y_pred = np.argmax(model.predict(test_data), axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv("outputs/classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)\
        .to_csv("outputs/confusion_matrix.csv")

    with mlflow.start_run(nested=True):
        mlflow.log_artifact("outputs/classification_report.csv")
        mlflow.log_artifact("outputs/confusion_matrix.csv")

        mlflow.log_metric("test_accuracy", report["accuracy"])

    logger.info("Evaluation completed")
