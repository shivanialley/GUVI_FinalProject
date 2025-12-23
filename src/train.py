import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.data_loader import load_data
from src.preprocessing import build_preprocessor
from src.feature_engineering import add_features


MODEL_NAME = "BankTermDepositModel"


def train_model():
    # -----------------------------
    # MLflow setup
    # -----------------------------
    mlflow.set_experiment("Bank-Term-Deposit-Prediction")

    with mlflow.start_run(run_name="xgboost_smote"):

        # -----------------------------
        # 1. Load data
        # -----------------------------
        df = load_data("data/train.csv")
        df = add_features(df)

        # -----------------------------
        # 2. Target handling
        # -----------------------------
        df["y"] = df["y"].astype(str).str.strip().str.lower()

        if set(df["y"].unique()).issubset({"yes", "no"}):
            y = df["y"].map({"yes": 1, "no": 0})
        else:
            y = df["y"].astype(int)

        X = df.drop("y", axis=1)

        # -----------------------------
        # 3. Preprocessing
        # -----------------------------
        preprocessor = build_preprocessor(X)

        # -----------------------------
        # 4. Model
        # -----------------------------
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        )

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model)
        ])

        # -----------------------------
        # 5. Split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # -----------------------------
        # 6. Train
        # -----------------------------
        pipeline.fit(X_train, y_train)

        # -----------------------------
        # 7. Metrics
        # -----------------------------
        probs = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        # -----------------------------
        # 8. MLflow logging
        # -----------------------------
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.log_metric("roc_auc", auc)

        # -----------------------------
        # 9. Register model
        # -----------------------------
        model_uri = mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        ).model_uri

        # -----------------------------
        # 10. Save local copy (optional)
        # -----------------------------
        joblib.dump(pipeline, "models/model.pkl")

        print("✅ Model trained & registered")
        print("ROC-AUC:", auc)
        print("Registered as:", MODEL_NAME)

        return pipeline, X_test, y_test