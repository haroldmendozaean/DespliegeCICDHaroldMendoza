import os
import pandas as pd

def test_existe_dataset():
    assert os.path.exists('data/housing.csv'), 'No se encontró data/housing.csv'

def test_tiene_target():
    df = pd.read_csv('data/housing.csv')
    assert 'SalePrice' in df.columns, "El dataset no contiene la columna objetivo 'SalePrice'"
