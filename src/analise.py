from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix
)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")


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

# Separar os dados em variáveis independentes (X) e variável dependente (y)
X = dados.drop(columns="classe")
y = dados["classe"]

# Divisão dos dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDados de treinamento:", X_treino.shape)
print("Dados de teste:", X_teste.shape)

# Padronização dos dados
scaler = StandardScaler()
X_treino_padronizado = scaler.fit_transform(X_treino)
X_teste_padronizado = scaler.transform(X_teste)
print("\nDados de treinamento padronizados:", X_treino_padronizado.shape)
print("Dados de teste padronizados:", X_teste_padronizado.shape)

# Treinamento do modelo de Árvore de Decisão
modelo_arvore = DecisionTreeClassifier(
    random_state=42
)

modelo_arvore.fit(
    X_treino_padronizado,
    y_treino
)

# Avaliação do modelo de Árvore de Decisão
y_pred_arvore = modelo_arvore.predict(
    X_teste_padronizado
)

print("\nPredições da Árvore de Decisão:")
print(y_pred_arvore)

# Cálculo das métricas de avaliação do modelo de Árvore de Decisão
accuracy_arvore = accuracy_score(
    y_teste,
    y_pred_arvore
)

recall_arvore = recall_score(
    y_teste,
    y_pred_arvore,
    average="macro"
)

matriz_arvore = confusion_matrix(
    y_teste,
    y_pred_arvore
)

print("\n===== ÁRVORE DE DECISÃO =====")
print("Acurácia:", accuracy_arvore)
print("Recall médio:", recall_arvore)
print("\nMatriz de confusão:")
print(matriz_arvore)

# Gerar a matriz de confusão para o modelo de Árvore de Decisão
disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz_arvore,
    display_labels=["Classe 0", "Classe 1", "Classe 2"]
)

disp.plot()
plt.title("Matriz de Confusão - Árvore de Decisão")
plt.tight_layout()

plt.savefig(
    "graficos/matriz_arvore.png"
)

plt.close()