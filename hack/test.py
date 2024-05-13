import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from ml_kx_train import ml_kx_predict_and_to_dataframe


def to_binary(y_pred):
    # Reasigna los valores basados en la condición
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return y_pred


def calcular_metricas(y_real, y_pred, nombre_modelo):
    # Calcular la precisión (accuracy)
    precision = accuracy_score(y_real, y_pred)
    print(f"Accuracy del modelo {nombre_modelo}: {precision:.4f}")

    # Calcular el recall
    rec = recall_score(y_real, y_pred)
    print(f"Recall del modelo {nombre_modelo}: {rec:.4f} \n")


if __name__ == "__main__":
    # Cargar x_test e y_test desde los archivos CSV:
    x_test = pd.read_csv('data/x_test.csv', index_col=0)
    x_test_ban = pd.read_csv('data/x_test_ban.csv', index_col=0)
    y_test = pd.read_csv('data/y_test.csv', index_col=0)

    # Cargar los modelos aprendidos:
    with open('pickle/reg_model_ml_ox.pkl', 'rb') as file:
        loaded_reg_model_ml_ox = pickle.load(file)
    with open('pickle/reg_model_ml_ban.pkl', 'rb') as file:
        loaded_reg_model_ml_ban = pickle.load(file)

    # Hacer las predicciones:
    y_pred_ml_ox = loaded_reg_model_ml_ox.predict(x_test)
    y_pred_ml_ban = loaded_reg_model_ml_ban.predict(x_test_ban)
    y_pred_biased_mlx = loaded_reg_model_ml_ox.predict(ml_kx_predict_and_to_dataframe(x_test_ban))

    # Las convertimos en binarios para la clasificación:
    y_pred_ml_ox = to_binary(y_pred_ml_ox)
    y_pred_ml_ban = to_binary(y_pred_ml_ban)
    y_pred_biased_mlx = to_binary(y_pred_biased_mlx)

    # Calcular las métricas:
    calcular_metricas(y_test, y_pred_ml_ox, "Modelo ML OX")
    calcular_metricas(y_test, y_pred_ml_ban, "Modelo ML BAN")
    calcular_metricas(y_test, y_pred_biased_mlx, "Modelo Biased MLX")

