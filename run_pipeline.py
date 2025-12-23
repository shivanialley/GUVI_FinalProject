from src.train import train_model
from src.evaluate import evaluate_model
from src.logger import logger

if __name__ == "__main__":
    logger.info("PIPELINE STARTED")
    print("🚀 Starting COVID-19 X-ray Pipeline")

    train_model()
    evaluate_model()

    logger.info("PIPELINE COMPLETED")
    print("✅ Pipeline completed successfully")
