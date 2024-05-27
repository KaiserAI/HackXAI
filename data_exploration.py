import pandas as pd
from matplotlib import pyplot as plt


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

# Contar los ejemplos de cada categoría de sexo
conteo_sex_por_categoria = data[bias_category_name].value_counts()

# Filtrar los datos por bias_category_name y contar los valores del target para cada categoría
conteo_approval_por_categoria = data.groupby(bias_category_name)[target].value_counts()

# Imprimir los resultados
print("Conteo de ejemplos por sexo:")
print(conteo_sex_por_categoria)

print("Conteo de valores de approval por categoría de sexo:")
print(conteo_approval_por_categoria)

corr_mat = data.corr()
print(corr_mat["Creditworthiness"])
