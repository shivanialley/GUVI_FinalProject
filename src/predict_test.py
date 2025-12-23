import pandas as pd
import joblib
from src.feature_engineering import add_features


def run_predict():
    print("📦 Running batch prediction...")

    model = joblib.load("models/model.pkl")

    test_df = pd.read_csv("data/test.csv")
    test_df = add_features(test_df)

    preds = model.predict(test_df)
    probs = model.predict_proba(test_df)[:, 1]

    submission = pd.DataFrame({
        "id": test_df.index,
        "y": preds
    })

    submission.to_csv("submission.csv", index=False)
    print("✅ submission.csv created")
