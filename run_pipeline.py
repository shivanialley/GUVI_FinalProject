import logging
import os
import argparse

from src.train import train_model
from src.evaluate import evaluate_model
from src.explain import run_explain
from src.predict_test import run_predict


# -----------------------------
# Logging setup
# -----------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(run_explain_flag: bool, run_predict_flag: bool):
    logging.info("🚀 Pipeline started")

    # -----------------------------
    # Train + Evaluate (ALWAYS)
    # -----------------------------
    model, X_test, y_test = train_model()
    logging.info("Training completed")

    evaluate_model(model, X_test, y_test)
    logging.info("Evaluation completed")

    # -----------------------------
    # Explainability (OPTIONAL)
    # -----------------------------
    if run_explain_flag:
        logging.info("Running SHAP explainability")
        run_explain()
        logging.info("SHAP explanation completed")

    # -----------------------------
    # Batch Prediction (OPTIONAL)
    # -----------------------------
    if run_predict_flag:
        logging.info("Running batch prediction")
        run_predict()
        logging.info("Batch prediction completed")

    logging.info("✅ Pipeline finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Pipeline Runner")

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Run SHAP explainability"
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run batch prediction on test data"
    )

    args = parser.parse_args()

    main(
        run_explain_flag=args.explain,
        run_predict_flag=args.predict
    )