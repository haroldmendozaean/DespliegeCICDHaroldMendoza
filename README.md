# Proyecto EAN – CI/CD de Machine Learning con MLflow y GitHub Actions (Precios de Vivienda)

Este proyecto implementa un pipeline completo de Machine Learning con **preprocesamiento**, **entrenamiento**, **evaluación** y **seguimiento con MLflow**, automatizado mediante **GitHub Actions** (CI/CD).

## Estructura
```
.
├── config.yaml
├── data/housing.csv
├── src/train.py
├── src/evaluate.py
├── tests/test_basic.py
├── .github/workflows/ml.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Ejecución local
```bash
make install
make train
make validate
```

## Configuración
Hiperparámetros, rutas y división de datos en `config.yaml`.

## CI/CD en GitHub
En cada push a `main`:
1. Linter (flake8)
2. Tests (pytest)
3. Entrenamiento (make train)
4. Validación (make validate)
5. Subida del artefacto `model.pkl`

## MLflow
Tracking local: `file://mlruns`. Se registran **parámetros**, **métricas**, **firma** y **ejemplo de entrada**.
