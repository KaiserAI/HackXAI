import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from train import LinearRegressionBinary


def calcular_metricas(y_real, y_pred, nombre_modelo):
    # Calcular la precisión (accuracy)
    precision = accuracy_score(y_real, y_pred)
    print(f"Accuracy del modelo {nombre_modelo}: {precision:.4f}")

    # Calcular el recall
    rec = recall_score(y_real, y_pred)
    print(f"Recall del modelo {nombre_modelo}: {rec:.4f} \n")


if __name__ == "__main__":
    # Cargar los archivos CSV:
    x_test = pd.read_csv('data/x_test.csv', index_col=0)
    x_without_bias_category_test = pd.read_csv('data/x_without_bias_category_test.csv', index_col=0)
    y_test = pd.read_csv('data/y_test.csv', index_col=0)

    # Cargar los modelos:
    loaded_reg_model_ml_ox_linear = LinearRegressionBinary()
    loaded_reg_model_ml_ox_linear.load_model("pickle/reg_model_ml_ox.pkl")

    loaded_reg_model_ml_ban_linear = LinearRegressionBinary()
    loaded_reg_model_ml_ban_linear.load_model("pickle/reg_model_ml_ban.pkl")

    loaded_reg_model_biased_mlx = LinearRegressionBinary()
    loaded_reg_model_biased_mlx.load_model("pickle/reg_model_biased_mlx.pkl")

    # Hacer las predicciones:
    y_pred_ml_ox_linear = loaded_reg_model_ml_ox_linear.predict(x_test)
    y_pred_ml_ban_linear = loaded_reg_model_ml_ban_linear.predict(x_without_bias_category_test)
    y_pred_biased_mlx_linear = loaded_reg_model_ml_ban_linear.predict(x_without_bias_category_test)

    # Calcular las métricas:
    calcular_metricas(y_test, y_pred_ml_ox_linear, "Modelo ML OX Linear")
    calcular_metricas(y_test, y_pred_ml_ban_linear, "Modelo ML BAN Linear")
    calcular_metricas(y_test, y_pred_biased_mlx_linear, "Modelo Biased MLX Linear")
