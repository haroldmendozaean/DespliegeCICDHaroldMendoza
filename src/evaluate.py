import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from mlflow.models import infer_signature

DATA_PATH = "data/housing.csv"
MODEL_PATH = "model.pkl"
MLRUNS_DIR = "mlruns"
EXPERIMENT_NAME = "Housing-Prices-CICD"

def main():
    print("=== Training pipeline (GradientBoosting) with MLflow ===")
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{os.path.abspath(MLRUNS_DIR)}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Metrics -> MSE: {mse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f}")

    with mlflow.start_run() as run:
        mlflow.log_param("algorithm", "GradientBoostingRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("max_depth", 3)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        signature = infer_signature(X_train, pipe.predict(X_train[:5]))
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )

    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
