import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def cargar_config(ruta: str = "config.yaml"):
    if os.path.exists(ruta):
        cfg_path = ruta
    elif os.path.exists("src/config.yaml"):
        cfg_path = "src/config.yaml"
    else:
        raise FileNotFoundError("No se encontró config.yaml (raíz o src/)")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    print("=== Validación de model.pkl en conjunto de prueba ===")
    cfg = cargar_config()
    datos_path = cfg["rutas"]["datos"]
    modelo_path = cfg["rutas"]["modelo"]
    test_size = cfg["division"]["test_size"]
    random_state = cfg["division"]["random_state"]

    df = pd.read_csv(datos_path)
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    modelo = joblib.load(modelo_path)
    preds = modelo.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"[VALIDACIÓN] MSE: {mse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f}")
    if r2 < 0.70:
        raise SystemExit("Falla de calidad: R2 por debajo del umbral (0.70).")

if __name__ == "__main__":
    main()
