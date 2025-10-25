import os
import yaml
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

def cargar_config(ruta: str = "config.yaml"):
    if os.path.exists(ruta):
        cfg_path = ruta
    elif os.path.exists("src/config.yaml"):
        cfg_path = "src/config.yaml"
    else:
        raise FileNotFoundError("No se encontró config.yaml (raíz o src/)")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def construir_modelo(cfg):
    alg = cfg["modelo"]["algoritmo"]
    params = cfg["modelo"]["parametros"]
    if alg != "GradientBoostingRegressor":
        raise ValueError(f"Plantilla soporta GradientBoostingRegressor. Recibido: {alg}")
    return GradientBoostingRegressor(**params, random_state=cfg["division"]["random_state"])

def main():
    print("=== Entrenamiento con MLflow (GradientBoosting) ===")
    cfg = cargar_config()
    datos_path = cfg["rutas"]["datos"]
    modelo_path = cfg["rutas"]["modelo"]
    mlruns_dir = cfg["rutas"]["mlruns"]
    exp_nombre = cfg["experimento"]["nombre"]
    test_size = cfg["division"]["test_size"]
    random_state = cfg["division"]["random_state"]

    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
    mlflow.set_experiment(exp_nombre)

    df = pd.read_csv(datos_path)
    if "SalePrice" not in df.columns:
        raise ValueError("La columna objetivo 'SalePrice' no existe en el dataset.")
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    modelo = construir_modelo(cfg)
    pipeline = Pipeline(steps=[("prep", preprocesador), ("modelo", modelo)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Métricas -> MSE: {mse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f}")

    with mlflow.start_run() as run:
        mlflow.log_param("algoritmo", cfg["modelo"]["algoritmo"])
        for k, v in cfg["modelo"]["parametros"].items():
            mlflow.log_param(k, v)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        firma = infer_signature(X_train, pipeline.predict(X_train[:5]))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=firma,
            input_example=X_train.iloc[:5]
        )

    joblib.dump(pipeline, modelo_path)
    print(f"✅ Modelo guardado en {modelo_path}")

if __name__ == "__main__":
    main()
