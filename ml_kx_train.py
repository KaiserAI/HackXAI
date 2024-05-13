import pickle

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/german_data.csv', sep=',', decimal='.', index_col=0, na_values='null')
x_ml_kx = data.drop(labels=['sex', 'approval'], axis=1).copy()
y_ml_kx = data["sex"].copy()

# Modelo ml_kx: entrenado para inferir la categoría bias.
random_state = 286
x_train_ml_kx, x_test_ml_kx = train_test_split(x_ml_kx, test_size=0.2, random_state=random_state, stratify=y_ml_kx)
y_train_ml_kx, y_test_ml_kx = train_test_split(y_ml_kx, test_size=0.2, random_state=random_state, stratify=y_ml_kx)

# Ordenar los df por id y guardarlos en csv:
x_train_ml_kx.sort_index(inplace=True)
y_train_ml_kx.sort_index(inplace=True)
x_test_ml_kx.to_csv('data/x_test_ml_kx.csv', index=True)
y_test_ml_kx.to_csv('data/y_test_ml_kx.csv', index=True)

reg_model_ml_kx = linear_model.LinearRegression()
reg_model_ml_kx.fit(x_train_ml_kx, y_train_ml_kx)
with open('pickle/reg_model_ml_kx.pkl', 'wb') as file:
    pickle.dump(reg_model_ml_kx, file)


def ml_kx_predict_and_to_dataframe(x_test_ml_kx_to_predict, reg_model_ml_kx_trained=reg_model_ml_kx):
    # Predice utilizando el modelo cargado
    y_score_model_ml_kx = reg_model_ml_kx_trained.predict(x_test_ml_kx_to_predict)

    # Reasigna los valores basados en la condición
    y_score_model_ml_kx = np.where(y_score_model_ml_kx > 0.5, 1, 0)

    # Convierte a DataFrame
    y_score_model_ml_kx = pd.DataFrame(y_score_model_ml_kx, columns=['sex'])  # es la columna 10

    # Restablece los índices de los DataFrames
    x_test_ml_kx_to_predict.reset_index(drop=True, inplace=True)
    y_score_model_ml_kx.reset_index(drop=True, inplace=True)

    nueva_columna = 8  # La columna donde se pondrá el atributo aprendido.
    x_test_ml_kx_to_predict = pd.concat([x_test_ml_kx_to_predict.iloc[:, :nueva_columna], y_score_model_ml_kx,
                                         x_test_ml_kx_to_predict.iloc[:, nueva_columna:]], axis=1)
    return x_test_ml_kx_to_predict
