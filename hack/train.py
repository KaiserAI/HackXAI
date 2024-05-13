import pickle
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# biased_mlx clasificador doble
# ml_ox entrenado con todo
# ml_kx entrenado pa aprender el sexo
# ml_ban

data = pd.read_csv('data/german_data.csv', sep=',', decimal='.', index_col=0, na_values='null')

# Separo el target de los datos de entreno.
y_data = data["approval"].copy()
x_data = data.drop('approval', axis=1).copy()
x_ban = x_data.drop(labels='sex', axis=1).copy()

# Separo en train y test:
random_state = 286
# Modelo ML_ox.
x_train, x_test = train_test_split(x_data, test_size=0.2, random_state=random_state, stratify=y_data)
y_train, y_test = train_test_split(y_data, test_size=0.2, random_state=random_state, stratify=y_data)
# Modelo ml_ban: entrenado sin la categoría bias.
x_train_ban, x_test_ban = train_test_split(x_ban, test_size=0.2, random_state=random_state, stratify=y_data)

# Ordenar los df por id y guardarlos en csv:
x_train.sort_index(inplace=True)
y_train.sort_index(inplace=True)
x_test.to_csv('data/x_test.csv', index=True)
x_test_ban.to_csv('data/x_test_ban.csv', index=True)
y_test.to_csv('data/y_test.csv', index=True)

# Entrenar los modelos
reg_model_ml_ox = linear_model.LinearRegression()
reg_model_ml_ox.fit(x_train, y_train)
with open('pickle/reg_model_ml_ox.pkl', 'wb') as file:
    pickle.dump(reg_model_ml_ox, file)

reg_model_ml_ban = linear_model.LinearRegression()
reg_model_ml_ban.fit(x_train_ban, y_train)
with open('pickle/reg_model_ml_ban.pkl', 'wb') as file:
    pickle.dump(reg_model_ml_ban, file)
