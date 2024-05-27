import numpy as np
import pickle
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# biased_mlx clasificador doble
# ml_ox entrenado con todo
# ml_kx entrenado pa aprender el sexo
# ml_ban

# [] Probar con svm, árboles y modelos probabilísticos.
class BinaryClassifier:
    def __init__(self, model):
        self.model = model

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, file_path):
        with open(file_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as model_file:
            self.model = pickle.load(model_file)


class LinearRegressionBinary(BinaryClassifier):
    def __init__(self, model=None):
        if model is None:
            model = linear_model.LinearRegression()
        super().__init__(model)

    def predict(self, x):
        predictions = super().predict(x)
        return np.where(predictions > 0.5, 1, 0)


class BiasedMlx(LinearRegressionBinary):
    def __init__(self, bias_category_tag, bias_category_pos, missing_feature_model, final_prediction_model):
        self.bias_category_tag = bias_category_tag
        self.bias_category_pos = bias_category_pos
        self.missing_feature_model = missing_feature_model
        super().__init__(final_prediction_model)

    def predict(self, x):
        # Predecir la categoría faltante
        y_score_ml_kx = self.missing_feature_model.predict(x)

        # Convertir a DataFrame
        y_score_ml_kx = pd.DataFrame(y_score_ml_kx, columns=[self.bias_category_tag])

        # Restablece los índices de los DataFrames
        x.reset_index(drop=True, inplace=True)
        y_score_ml_kx.reset_index(drop=True, inplace=True)

        # Concatenar la predicción con el DataFrame original
        x_with_bias_category_inferred = pd.concat([x.iloc[:, :self.bias_category_pos], y_score_ml_kx,
                                                   x.iloc[:, self.bias_category_pos:]], axis=1)

        # Realizar la predicción final utilizando el nuevo DataFrame
        predictions = super().predict(x_with_bias_category_inferred)
        return predictions


if __name__ == "__main__":
    data = pd.read_csv('data/german_data.csv', sep=',', decimal='.', index_col=0, na_values='null')
    bias_category_name = "sex"
    target = "approval"
    '''
    data = pd.read_csv('data/taiwan_data.csv', sep=',', decimal='.', index_col=0, na_values='null')
    bias_category_name = "SEX"
    target = "Creditworthiness"
    '''
    '''
    data = pd.read_csv('data/taiwan_data_02.csv', sep=';', decimal='.', index_col=0, na_values='null')
    bias_category_name = "PAY_0"  # "sex" "Age Group" "SEX"
    target = "Creditworthiness"
    '''

    bias_category_position = data.columns.get_loc(bias_category_name)

    # Particion:
    seed = 289
    train, test = train_test_split(data, test_size=0.2, random_state=seed)

    y_train = train[target].copy()
    y_test = test[target].copy()
    x_train = train.drop(labels=target, axis=1).copy()
    x_test = test.drop(labels=target, axis=1).copy()
    x_without_bias_category_train = train.drop(labels=[bias_category_name, target], axis=1).copy()
    x_without_bias_category_test = test.drop(labels=[bias_category_name, target], axis=1).copy()
    y_predict_bias_category_train = train[bias_category_name].copy()
    y_predict_bias_category_test = test[bias_category_name].copy()
    x_predict_bias_category_train = train.drop(labels=[bias_category_name, target], axis=1).copy()
    x_predict_bias_category_test = test.drop(labels=[bias_category_name, target], axis=1).copy()

    # Ordenar los df por id y guardarlos en csv:
    x_train.sort_index(inplace=True)
    y_train.sort_index(inplace=True)
    x_test.sort_index(inplace=True)
    y_test.sort_index(inplace=True)
    x_without_bias_category_train.sort_index(inplace=True)
    x_without_bias_category_test.sort_index(inplace=True)
    y_predict_bias_category_train.sort_index(inplace=True)
    y_predict_bias_category_test.sort_index(inplace=True)
    x_predict_bias_category_train.sort_index(inplace=True)
    x_predict_bias_category_test.sort_index(inplace=True)

    x_test.to_csv('data/x_test.csv', index=True)
    y_test.to_csv('data/y_test.csv', index=True)
    x_without_bias_category_test.to_csv('data/x_without_bias_category_test.csv', index=True)
    x_predict_bias_category_test.to_csv('data/x_predict_bias_category_test.csv', index=True)
    y_predict_bias_category_test.to_csv('data/y_predict_bias_category_test.csv', index=True)

    # Entrenar los modelos de regresión lineal
    reg_model_ml_ox = LinearRegressionBinary()
    reg_model_ml_ox.fit(x_train, y_train)
    reg_model_ml_ox.save_model("pickle/reg_model_ml_ox.pkl")

    reg_model_ml_ban = LinearRegressionBinary()
    reg_model_ml_ban.fit(x_without_bias_category_train, y_train)
    reg_model_ml_ban.save_model("pickle/reg_model_ml_ban.pkl")

    reg_model_ml_kx = LinearRegressionBinary()
    reg_model_ml_kx.fit(x_predict_bias_category_train, y_predict_bias_category_train)
    reg_model_ml_kx.save_model("pickle/reg_model_ml_kx.pkl")

    # Crear el modelo biased_mlx:
    reg_model_biased_mlx = BiasedMlx(bias_category_name, bias_category_position, reg_model_ml_kx, reg_model_ml_ox)
    reg_model_biased_mlx.save_model("pickle/reg_model_biased_mlx.pkl")
