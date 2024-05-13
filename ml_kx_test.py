import pickle

import numpy as np
import pandas as pd
from test import to_binary, calcular_metricas


# Cargar x_test e y_test desde los archivos CSV:
x_test_ml_kx = pd.read_csv('data/x_test_ml_kx.csv', index_col=0)
y_test_ml_kx = pd.read_csv('data/y_test_ml_kx.csv', index_col=0)

with open('pickle/reg_model_ml_kx.pkl', 'rb') as file:
    loaded_reg_model_ml_kx = pickle.load(file)

# Hacer las predicciones:
y_pred_ml_kx = loaded_reg_model_ml_kx.predict(x_test_ml_kx)
y_pred_ml_kx = to_binary(y_pred_ml_kx)
calcular_metricas(y_test_ml_kx, y_pred_ml_kx, "Modelo ML KX")
