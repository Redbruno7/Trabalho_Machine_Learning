import os
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


os.system('cls')

wine = load_wine()

dados = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

# Adicionar a coluna de classe ao DataFrame
dados["classe"] = wine.target
print(dados.head())

# Exibir informações sobre o DataFrame
print()
print("Quantidade de linhas:", dados.shape[0])
print("Quantidade de colunas:", dados.shape[1])

# Exibir os nomes das colunas do DataFrame
print()
print(dados.columns)

# Exibir informações detalhadas sobre o DataFrame
print()
print(dados.info())

# Exibir estatísticas descritivas do DataFrame
print()
print(dados.describe())

# Verificar a presença de valores nulos no DataFrame
print()
print(dados.isnull().sum())

# Verificar a presença de valores duplicados no DataFrame
print()
print("Duplicados:", dados.duplicated().sum())

# Exibir a contagem de cada classe no DataFrame
print(dados["classe"].value_counts())

# Gerar um gráfico de barras para a distribuição das classes
dados["classe"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Distribuição das classes")
plt.xlabel("Classe")
plt.ylabel("Quantidade de amostras")
plt.tight_layout()
plt.savefig("graficos/distribuicao_classes.png")
plt.close()

# Gerar um boxplot para identificar outliers nos atributos do DataFrame
plt.figure(figsize=(14, 7))
dados.drop(columns="classe").boxplot()
plt.title("Distribuição dos atributos e identificação de outliers")
plt.xlabel("Atributos")
plt.ylabel("Valores")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graficos/outliers.png")
plt.close()

# Gerar uma matriz de correlação entre os atributos do DataFrame
correlacao = dados.drop(columns="classe").corr()
plt.figure(figsize=(12, 10))
plt.imshow(correlacao, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Correlação")

plt.xticks(
    range(len(correlacao.columns)),
    correlacao.columns,
    rotation=90
)

plt.yticks(
    range(len(correlacao.columns)),
    correlacao.columns
)

plt.title("Matriz de correlação entre os atributos")
plt.tight_layout()
plt.savefig("graficos/correlacao.png")
plt.close()

# Gerar histogramas para cada atributo do DataFrame
dados.drop(columns="classe").hist(
    figsize=(14, 10),
    bins=15
)

plt.suptitle("Distribuição dos atributos do Wine Dataset")
plt.tight_layout()
plt.savefig("graficos/distribuicao_atributos.png")
plt.close()

# Separar os atributos (X) e a variável alvo (y)
X = dados.drop(columns="classe")
y = dados["classe"]

# Padronizar os dados utilizando o StandardScaler
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X)
print()
print("\nFormato dos dados padronizados:", X_padronizado.shape)