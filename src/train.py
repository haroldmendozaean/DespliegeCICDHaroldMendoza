import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import traceback
import sys

print("=== 🚀 INICIO ENTRENAMIENTO MODELO (MLflow CI/CD) ===")
print(f"Directorio actual de trabajo: {os.getcwd()}")

# --- Configuración de MLflow ---
mlruns_dir = os.path.join(os.getcwd(), "mlruns")
os.makedirs(mlruns_dir, exist_ok=True)

tracking_uri = f"file://{mlruns_dir}"
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "CI-CD-Lab2"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

print(f"Experimento activo: {experiment_name}")
print(f"Tracking URI: {tracking_uri}")
print(f"Ubicación artefactos: {experiment.artifact_location}")

# --- Entrenamiento del modelo ---
try:
    print("📊 Cargando dataset...")
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("🧠 Entrenando modelo LinearRegression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    print(f"✅ Modelo entrenado. MSE: {mse:.4f}")

    # --- Registrar el modelo en MLflow ---
    with mlflow.start_run() as run:
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"🗂️ Modelo loggeado en run_id: {run.info.run_id}")

    # --- Guardar copia local para el paso de validación ---
    model_path = os.path.join(os.getcwd(), "model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Modelo guardado correctamente en: {model_path}")

except Exception as e:
    print("❌ ERROR durante el entrenamiento o registro del modelo:")
    traceback.print_exc()
    sys.exit(1)

print("=== ✅ FIN DEL ENTRENAMIENTO Y REGISTRO DEL MODELO ===")
