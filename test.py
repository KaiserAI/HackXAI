import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from train import ml_kx_predict_and_to_dataframe
from train import BinaryClassifier


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
    x_without_bias_category_test = pd.read_csv('data/x_without_bias_category_test.csv', index_col=0)
    y_test = pd.read_csv('data/y_test.csv', index_col=0)

    bias_category = "SEX"  # "sex" "Age Group" "SEX"
    bias_category_position = x_test.columns.get_loc(bias_category)

    # Cargar los modelos lineales:
    loaded_reg_model_ml_ox_linear = BinaryClassifier()
    loaded_reg_model_ml_ox_linear.load_model("pickle/reg_model_ml_ox_linear.pkl")

    loaded_reg_model_ml_ban_linear = BinaryClassifier()
    loaded_reg_model_ml_ban_linear.load_model("pickle/reg_model_ml_ban_linear.pkl")

    # Cargar los modelos logisticos:
    loaded_reg_model_ml_ox_logistic = BinaryClassifier()
    loaded_reg_model_ml_ox_logistic.load_model("pickle/reg_model_ml_ox_logistic.pkl")

    loaded_reg_model_ml_ban_logistic = BinaryClassifier()
    loaded_reg_model_ml_ban_logistic.load_model("pickle/reg_model_ml_ban_logistic.pkl")

    # Hacer las predicciones:
    y_pred_ml_ox_linear = loaded_reg_model_ml_ox_linear.predict(x_test)
    y_pred_ml_ban_linear = loaded_reg_model_ml_ban_linear.predict(x_without_bias_category_test)
    y_pred_biased_mlx_linear = loaded_reg_model_ml_ox_linear.predict(ml_kx_predict_and_to_dataframe(
        "linear", x_without_bias_category_test, bias_category, bias_category_position))

    y_pred_ml_ox_logistic = loaded_reg_model_ml_ox_logistic.predict(x_test)
    y_pred_ml_ban_logistic = loaded_reg_model_ml_ban_logistic.predict(x_without_bias_category_test)
    y_pred_biased_mlx_logistic = loaded_reg_model_ml_ox_logistic.predict(ml_kx_predict_and_to_dataframe(
        "logistic", x_without_bias_category_test, bias_category, bias_category_position))

    # Calcular las métricas:
    calcular_metricas(y_test, y_pred_ml_ox_linear, "Modelo ML OX Linear")
    calcular_metricas(y_test, y_pred_ml_ban_linear, "Modelo ML BAN Linear")
    calcular_metricas(y_test, y_pred_biased_mlx_linear, "Modelo Biased MLX Linear")

    calcular_metricas(y_test, y_pred_ml_ox_logistic, "Modelo ML OX Logistic")
    calcular_metricas(y_test, y_pred_ml_ban_logistic, "Modelo ML BAN Logistic")
    calcular_metricas(y_test, y_pred_biased_mlx_logistic, "Modelo Biased MLX Logistic")
