import numpy as np
import pickle
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# biased_mlx clasificador doble
# ml_ox entrenado con todo
# ml_kx entrenado pa aprender el sexo
# ml_ban

# [] Dividir train y test antes de usos.
# [] Probar con svm, árboles y modelos probabilísticos.

class BinaryClassifier:
    def __init__(self, model_type='linear'):
        if model_type == 'linear':
            self.model = linear_model.LinearRegression()
        elif model_type == 'logistic':
            self.model = linear_model.LogisticRegression(max_iter=10000000)  # 1000 10000000
        else:
            raise ValueError("Unsupported model_type. Use 'linear' or 'logistic'.")
        self.model_type = model_type

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        if self.model_type == 'linear':
            predictions = self.model.predict(x)
            return np.where(predictions > 0.5, 1, 0)
        elif self.model_type == 'logistic':
            return self.model.predict(x)

    def save_model(self, file_path):
        with open(file_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as model_file:
            self.model = pickle.load(model_file)


def ml_kx_predict_and_to_dataframe(model_type, x_without_bias_category, bias_category_name, bias_category_pos):
    loaded_reg_model_kx = BinaryClassifier()
    if model_type == 'linear':
        loaded_reg_model_kx.load_model("pickle/reg_model_ml_kx_linear.pkl")
    elif model_type == 'logistic':
        loaded_reg_model_kx.load_model("pickle/reg_model_ml_kx_logistic.pkl")
    else:
        raise ValueError("Unsupported model_type. Use 'linear' or 'logistic'.")
    y_score_ml_kx = loaded_reg_model_kx.predict(x_without_bias_category)

    # Convierte a DataFrame
    y_score_ml_kx = pd.DataFrame(y_score_ml_kx, columns=[bias_category_name])
    # Restablece los índices de los DataFrames
    x_without_bias_category.reset_index(drop=True, inplace=True)
    y_score_ml_kx.reset_index(drop=True, inplace=True)

    x_with_bias_category_inferred = pd.concat([x_without_bias_category.iloc[:, :bias_category_pos], y_score_ml_kx,
                                               x_without_bias_category.iloc[:, bias_category_pos:]], axis=1)
    return x_with_bias_category_inferred


if __name__ == "__main__":
    # data = pd.read_csv('data/german_data.csv', sep=',', decimal='.', index_col=0, na_values='null')
    data = pd.read_csv('data/taiwan_data.csv', sep=',', decimal='.', index_col=0, na_values='null')

    # bias category
    bias_category = "SEX"  # "sex" "Age Group" "SEX"
    bias_category_position = data.columns.get_loc(bias_category)

    # target
    target = "Creditworthiness"  # "approval" "Creditworthiness"

    # Separo el target de los datos de entreno.
    y_data = data[target].copy()
    x_data = data.drop(labels=target, axis=1).copy()
    x_without_bias_category_data = data.drop(labels=[bias_category, target], axis=1).copy()
    y_predict_bias_category_data = data[bias_category].copy()
    x_predict_bias_category_data = data.drop(labels=[bias_category, target], axis=1).copy()

    # Separo en train y test:
    seed = 286
    # Modelo ML_ox.
    x_train, x_test = train_test_split(x_data, test_size=0.2, random_state=seed, stratify=y_data)
    y_train, y_test = train_test_split(y_data, test_size=0.2, random_state=seed, stratify=y_data)
    # Modelo ml_ban: entrenado sin la categoría bias.
    x_without_bias_category_train, x_without_bias_category_test = train_test_split(x_without_bias_category_data,
                                                                                   test_size=0.2, random_state=seed, stratify=y_data)
    # Modelo ml_kx: predice la categoría bias.
    x_predict_bias_category_train, x_predict_bias_category_test = train_test_split(x_predict_bias_category_data,
                                                                                   test_size=0.2, random_state=seed, stratify=y_predict_bias_category_data)
    y_predict_bias_category_train, y_predict_bias_category_test = train_test_split(y_predict_bias_category_data,
                                                                                   test_size=0.2, random_state=seed, stratify=y_predict_bias_category_data)

    # Ordenar los df por id y guardarlos en csv:
    # x_train.sort_index(inplace=True) # si quito estos 2 se va a la mierda todo.
    # y_train.sort_index(inplace=True)

    x_test.to_csv('data/x_test.csv', index=True)
    y_test.to_csv('data/y_test.csv', index=True)
    x_without_bias_category_test.to_csv('data/x_without_bias_category_test.csv', index=True)
    x_predict_bias_category_test.to_csv('data/x_predict_bias_category_test.csv', index=True)
    y_predict_bias_category_test.to_csv('data/y_predict_bias_category_test.csv', index=True)

    # Entrenar los modelos de regresión lineal
    reg_model_ml_ox_linear = BinaryClassifier(model_type='linear')
    reg_model_ml_ox_linear.fit(x_train, y_train)
    reg_model_ml_ox_linear.save_model("pickle/reg_model_ml_ox_linear.pkl")

    reg_model_ml_ban_linear = BinaryClassifier(model_type='linear')
    reg_model_ml_ban_linear.fit(x_without_bias_category_train, y_train)
    reg_model_ml_ban_linear.save_model("pickle/reg_model_ml_ban_linear.pkl")

    reg_model_ml_kx_linear = BinaryClassifier(model_type='linear')
    reg_model_ml_kx_linear.fit(x_predict_bias_category_train, y_predict_bias_category_train)
    reg_model_ml_kx_linear.save_model("pickle/reg_model_ml_kx_linear.pkl")

    # Entrenar los modelos de regresión logística
    reg_model_ml_ox_logistic = BinaryClassifier(model_type='logistic')
    reg_model_ml_ox_logistic.fit(x_train, y_train)
    reg_model_ml_ox_logistic.save_model("pickle/reg_model_ml_ox_logistic.pkl")

    reg_model_ml_ban_logistic = BinaryClassifier(model_type='logistic')
    reg_model_ml_ban_logistic.fit(x_without_bias_category_train, y_train)
    reg_model_ml_ban_logistic.save_model("pickle/reg_model_ml_ban_logistic.pkl")

    reg_model_ml_kx_logistic = BinaryClassifier(model_type='logistic')
    reg_model_ml_kx_logistic.fit(x_predict_bias_category_train, y_predict_bias_category_train)
    reg_model_ml_kx_logistic.save_model("pickle/reg_model_ml_kx_logistic.pkl")
