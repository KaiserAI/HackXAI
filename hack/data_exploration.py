import pandas as pd

data = pd.read_csv('data/german_data.csv', sep=',', decimal='.', index_col=0, na_values='null')

# Contar los ejemplos de cada categoría de sexo
conteo_sex_por_categoria = data['sex'].value_counts()

# Filtrar los datos por sexo y contar los valores de approval para cada categoría de sexo
conteo_approval_por_categoria = data.groupby('sex')['approval'].value_counts()

# Imprimir los resultados
print("Conteo de ejemplos por sexo:")
print(conteo_sex_por_categoria)

print("Conteo de valores de approval por categoría de sexo:")
print(conteo_approval_por_categoria)
