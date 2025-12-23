import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.feature_engineering import add_features


def run_explain():
    """
    Runs SHAP explainability on trained model.
    Called ONLY from run_pipeline.py when --explain flag is used.
    """

    print("🔍 Running SHAP explainability...")

    # -----------------------------
    # Load trained pipeline
    # -----------------------------
    pipeline = joblib.load("models/model.pkl")

    preprocessor = pipeline.named_steps["preprocess"]
    xgb_model = pipeline.named_steps["model"]

    # -----------------------------
    # Load and prepare data
    # -----------------------------
    df = load_data("data/train.csv")
    df = add_features(df)

    X = df.drop("y", axis=1)

    # Small sample (SHAP speed)
    X_sample = X.sample(50, random_state=42)

    # -----------------------------
    # Preprocess features
    # -----------------------------
    X_transformed = preprocessor.transform(X_sample)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    # -----------------------------
    # Prediction function (CRITICAL)
    # -----------------------------
    def predict_fn(X):
        """
        X is already preprocessed numeric data
        """
        return xgb_model.predict_proba(X)

    # -----------------------------
    # SHAP (MODEL-AGNOSTIC & SAFE)
    # -----------------------------
    explainer = shap.Explainer(
        predict_fn,
        X_transformed_df,
        algorithm="permutation"
    )

    shap_values = explainer(X_transformed_df)

    # -----------------------------
    # Plot (positive class)
    # -----------------------------
    shap.summary_plot(
        shap_values[:, :, 1],
        X_transformed_df,
        show=True
    )

    plt.show()