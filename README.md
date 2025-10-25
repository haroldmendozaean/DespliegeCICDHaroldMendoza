# Housing Prices CI/CD with MLflow + GitHub Actions

This repository demonstrates a full ML pipeline using a **public housing dataset**,
a **Gradient Boosting Regressor**, and **MLflow** for experiment tracking.
The pipeline is automated with **GitHub Actions**.

## Project structure
```
.
├── .github/workflows/ml.yml       # CI/CD workflow
├── data/housing.csv               # dataset (synthetic Kaggle-like)
├── src/train.py                   # training + MLflow logging
├── src/evaluate.py                # evaluation + quality gate
├── Makefile                       # convenience tasks
├── requirements.txt
└── mlruns/                        # local MLflow tracking (created at runtime)
```

## Quick start (local)
```bash
make install
make train
make validate
```

## CI/CD
On each push to `main`, the workflow:
1. Installs dependencies
2. Trains the model
3. Validates performance
4. Uploads `model.pkl` as a build artifact
