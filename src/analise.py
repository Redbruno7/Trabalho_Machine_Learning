from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    silhouette_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use("Agg")


# ============================================================
# FUNÇÃO PARA CALCULAR A ESPECIFICIDADE
# ============================================================

def calcular_especificidade(y_real, y_pred):
    matriz = confusion_matrix(y_real, y_pred)
    especificidades = []

    for i in range(len(matriz)):
        verdadeiro_negativo = matriz.sum() - (
            matriz[i, :].sum()
            + matriz[:, i].sum()
            - matriz[i, i]
        )

        falso_positivo = matriz[:, i].sum() - matriz[i, i]

        especificidade = verdadeiro_negativo / (
            verdadeiro_negativo + falso_positivo
        )

        especificidades.append(especificidade)
    return especificidades


# ============================================================
# 1. PREPARAÇÃO DO AMBIENTE
# ============================================================
os.system("cls")

# Criar a pasta de gráficos caso ela ainda não exista
os.makedirs("graficos", exist_ok=True)


# ============================================================
# 2. CARREGAMENTO DO WINE DATASET
# ============================================================
wine = load_wine()

dados = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

# Adicionar a classe ao DataFrame
dados["classe"] = wine.target

print("================================")
print("       WINE DATASET")
print("================================")
print("\nPrimeiras amostras:")
print(dados.head())


# ============================================================
# 3. ANÁLISE INICIAL DOS DADOS
# ============================================================

print("\n================================")
print("     ANÁLISE DOS DADOS")
print("================================")
print("\nQuantidade de linhas:", dados.shape[0])
print("Quantidade de colunas:", dados.shape[1])
print("\nNomes das colunas:")
print(dados.columns)
print("\nInformações do DataFrame:")
dados.info()
print("\nEstatísticas descritivas:")
print(dados.describe())
print("\nValores nulos por coluna:")
print(dados.isnull().sum())
print("\nQuantidade de dados duplicados:")
print(dados.duplicated().sum())
print("\nQuantidade de amostras por classe:")
print(dados["classe"].value_counts().sort_index())


# ============================================================
# 4. ANÁLISE EXPLORATÓRIA — DISTRIBUIÇÃO DAS CLASSES
# ============================================================

dados["classe"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Distribuição das classes")
plt.xlabel("Classe")
plt.ylabel("Quantidade de amostras")
plt.tight_layout()

plt.savefig(
    "graficos/distribuicao_classes.png"
)

plt.close()


# ============================================================
# 5. ANÁLISE EXPLORATÓRIA — OUTLIERS
# ============================================================
plt.figure(figsize=(14, 7))
dados.drop(columns="classe").boxplot()

plt.title(
    "Distribuição dos atributos e identificação de outliers"
)

plt.xlabel("Atributos")
plt.ylabel("Valores")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "graficos/outliers.png"
)

plt.close()


# ============================================================
# 6. ANÁLISE EXPLORATÓRIA — CORRELAÇÃO
# ============================================================
correlacao = dados.drop(columns="classe").corr()
plt.figure(figsize=(12, 10))

plt.imshow(
    correlacao,
    cmap="coolwarm",
    aspect="auto"
)

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

plt.title(
    "Matriz de correlação entre os atributos"
)

plt.tight_layout()

plt.savefig(
    "graficos/correlacao.png"
)

plt.close()


# ============================================================
# 7. ANÁLISE EXPLORATÓRIA — DISTRIBUIÇÃO DOS ATRIBUTOS
# ============================================================
dados.drop(columns="classe").hist(
    figsize=(14, 10),
    bins=15
)

plt.suptitle(
    "Distribuição dos atributos do Wine Dataset"
)

plt.tight_layout()

plt.savefig(
    "graficos/distribuicao_atributos.png"
)

plt.close()


# ============================================================
# 8. SEPARAÇÃO ENTRE ATRIBUTOS E CLASSE
# ============================================================
X = dados.drop(columns="classe")
y = dados["classe"]


# ============================================================
# 9. DIVISÃO DOS DADOS EM TREINAMENTO E TESTE
# ============================================================
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n================================")
print("     DIVISÃO DOS DADOS")
print("================================")

print("\nDados de treinamento:", X_treino.shape)
print("Dados de teste:", X_teste.shape)


# ============================================================
# 10. PADRONIZAÇÃO DOS DADOS
# ============================================================
scaler = StandardScaler()

X_treino_padronizado = scaler.fit_transform(
    X_treino
)

X_teste_padronizado = scaler.transform(
    X_teste
)

print(
    "\nDados de treinamento padronizados:",
    X_treino_padronizado.shape
)

print(
    "Dados de teste padronizados:",
    X_teste_padronizado.shape
)


# ============================================================
# 11. CONFIGURAÇÃO DOS MODELOS
# ============================================================

# Árvore de Decisão
pipeline_arvore = Pipeline([
    (
        "padronizacao",
        StandardScaler()
    ),
    (
        "modelo",
        DecisionTreeClassifier(
            random_state=42
        )
    )
])

# KNN
pipeline_knn = Pipeline([
    (
        "padronizacao",
        StandardScaler()
    ),
    (
        "modelo",
        KNeighborsClassifier(
            n_neighbors=5
        )
    )
])

# SVM
pipeline_svm = Pipeline([
    (
        "padronizacao",
        StandardScaler()
    ),
    (
        "modelo",
        CalibratedClassifierCV(
            SVC(
                kernel="rbf",
                random_state=42
            ),
            ensemble=False
        )
    )
])


# ============================================================
# 12. CONFIGURAÇÃO DO K-FOLD
# ============================================================
kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 13. VALIDAÇÃO K-FOLD — ÁRVORE DE DECISÃO
# ============================================================
scores_arvore = cross_val_score(
    pipeline_arvore,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("\n================================")
print(" K-FOLD - ÁRVORE DE DECISÃO")
print("================================")

print(
    "Acurácias de cada fold:",
    scores_arvore
)

print(
    "Acurácia média:",
    scores_arvore.mean()
)

print(
    "Desvio padrão:",
    scores_arvore.std()
)


# ============================================================
# 14. VALIDAÇÃO K-FOLD — KNN
# ============================================================
scores_knn = cross_val_score(
    pipeline_knn,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("\n================================")
print("          K-FOLD - KNN")
print("================================")

print(
    "Acurácias de cada fold:",
    scores_knn
)

print(
    "Acurácia média:",
    scores_knn.mean()
)

print(
    "Desvio padrão:",
    scores_knn.std()
)


# ============================================================
# 15. VALIDAÇÃO K-FOLD — SVM
# ============================================================
scores_svm = cross_val_score(
    pipeline_svm,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("\n================================")
print("          K-FOLD - SVM")
print("================================")

print(
    "Acurácias de cada fold:",
    scores_svm
)

print(
    "Acurácia média:",
    scores_svm.mean()
)

print(
    "Desvio padrão:",
    scores_svm.std()
)


# ============================================================
# 16. TREINAMENTO DOS MODELOS
# ============================================================
pipeline_arvore.fit(
    X_treino,
    y_treino
)

pipeline_knn.fit(
    X_treino,
    y_treino
)

pipeline_svm.fit(
    X_treino,
    y_treino
)


# ============================================================
# 17. PREDIÇÕES NO CONJUNTO DE TESTE
# ============================================================
y_pred_arvore = pipeline_arvore.predict(
    X_teste
)

y_pred_knn = pipeline_knn.predict(
    X_teste
)

y_pred_svm = pipeline_svm.predict(
    X_teste
)


# ============================================================
# 18. CÁLCULO DE ACCURACY E RECALL
# ============================================================
accuracy_arvore = accuracy_score(
    y_teste,
    y_pred_arvore
)

accuracy_knn = accuracy_score(
    y_teste,
    y_pred_knn
)

accuracy_svm = accuracy_score(
    y_teste,
    y_pred_svm
)

recall_arvore = recall_score(
    y_teste,
    y_pred_arvore,
    average="macro"
)

recall_knn = recall_score(
    y_teste,
    y_pred_knn,
    average="macro"
)

recall_svm = recall_score(
    y_teste,
    y_pred_svm,
    average="macro"
)


print("\n================================")
print("       MÉTRICAS DOS MODELOS")
print("================================")
print("\nÁrvore de Decisão")
print("Accuracy:", accuracy_arvore)
print("Recall:", recall_arvore)
print("\nKNN")
print("Accuracy:", accuracy_knn)
print("Recall:", recall_knn)
print("\nSVM")
print("Accuracy:", accuracy_svm)
print("Recall:", recall_svm)


# ============================================================
# 19. MATRIZES DE CONFUSÃO
# ============================================================
matriz_arvore = confusion_matrix(
    y_teste,
    y_pred_arvore
)

matriz_knn = confusion_matrix(
    y_teste,
    y_pred_knn
)

matriz_svm = confusion_matrix(
    y_teste,
    y_pred_svm
)

print("\n================================")
print("     MATRIZES DE CONFUSÃO")
print("================================")
print("\nÁrvore de Decisão:")
print(matriz_arvore)
print("\nKNN:")
print(matriz_knn)
print("\nSVM:")
print(matriz_svm)

# Matriz da Árvore
disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz_arvore,
    display_labels=[
        "Classe 0",
        "Classe 1",
        "Classe 2"
    ]
)

disp.plot()

plt.title(
    "Matriz de Confusão - Árvore de Decisão"
)

plt.tight_layout()

plt.savefig(
    "graficos/matriz_arvore.png"
)

plt.close()

# Matriz do KNN
disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz_knn,
    display_labels=[
        "Classe 0",
        "Classe 1",
        "Classe 2"
    ]
)

disp.plot()

plt.title(
    "Matriz de Confusão - KNN"
)

plt.tight_layout()

plt.savefig(
    "graficos/matriz_knn.png"
)

plt.close()

# Matriz do SVM
disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz_svm,
    display_labels=[
        "Classe 0",
        "Classe 1",
        "Classe 2"
    ]
)

disp.plot()

plt.title(
    "Matriz de Confusão - SVM"
)

plt.tight_layout()

plt.savefig(
    "graficos/matriz_svm.png"
)

plt.close()


# ============================================================
# 20. CÁLCULO DA ESPECIFICIDADE
# ============================================================
especificidade_arvore = calcular_especificidade(
    y_teste,
    y_pred_arvore
)

especificidade_knn = calcular_especificidade(
    y_teste,
    y_pred_knn
)

especificidade_svm = calcular_especificidade(
    y_teste,
    y_pred_svm
)

media_especificidade_arvore = np.mean(
    especificidade_arvore
)

media_especificidade_knn = np.mean(
    especificidade_knn
)

media_especificidade_svm = np.mean(
    especificidade_svm
)

print("\n================================")
print("       ESPECIFICIDADE")
print("================================")
print("\nÁrvore de Decisão:")

print(
    "Por classe:",
    especificidade_arvore
)

print(
    "Média:",
    media_especificidade_arvore
)

print("\nKNN:")

print(
    "Por classe:",
    especificidade_knn
)

print(
    "Média:",
    media_especificidade_knn
)

print("\nSVM:")

print(
    "Por classe:",
    especificidade_svm
)

print(
    "Média:",
    media_especificidade_svm
)


# ============================================================
# 21. CÁLCULO DA AUC
# ============================================================
prob_arvore = pipeline_arvore.predict_proba(
    X_teste
)

prob_knn = pipeline_knn.predict_proba(
    X_teste
)

prob_svm = pipeline_svm.predict_proba(
    X_teste
)

auc_arvore = roc_auc_score(
    y_teste,
    prob_arvore,
    multi_class="ovr",
    average="macro"
)

auc_knn = roc_auc_score(
    y_teste,
    prob_knn,
    multi_class="ovr",
    average="macro"
)

auc_svm = roc_auc_score(
    y_teste,
    prob_svm,
    multi_class="ovr",
    average="macro"
)

print("\n================================")
print("             AUC")
print("================================")

print(
    "Árvore de Decisão:",
    auc_arvore
)

print(
    "KNN:",
    auc_knn
)

print(
    "SVM:",
    auc_svm
)


# ============================================================
# 22. CURVAS ROC
# ============================================================
y_teste_binario = label_binarize(
    y_teste,
    classes=[0, 1, 2]
)

fpr_arvore = {}
tpr_arvore = {}

fpr_knn = {}
tpr_knn = {}

fpr_svm = {}
tpr_svm = {}

for i in range(3):
    fpr_arvore[i], tpr_arvore[i], _ = roc_curve(
        y_teste_binario[:, i],
        prob_arvore[:, i]
    )

    fpr_knn[i], tpr_knn[i], _ = roc_curve(
        y_teste_binario[:, i],
        prob_knn[:, i]
    )

    fpr_svm[i], tpr_svm[i], _ = roc_curve(
        y_teste_binario[:, i],
        prob_svm[:, i]
    )

plt.figure(figsize=(10, 7))

# Árvore de Decisão
for i in range(3):
    plt.plot(
        fpr_arvore[i],
        tpr_arvore[i],
        linestyle="--",
        label=f"Árvore - Classe {i}"
    )

# KNN
for i in range(3):
    plt.plot(
        fpr_knn[i],
        tpr_knn[i],
        linestyle="-.",
        label=f"KNN - Classe {i}"
    )

# SVM
for i in range(3):

    plt.plot(
        fpr_svm[i],
        tpr_svm[i],
        label=f"SVM - Classe {i}"
    )

# Classificador aleatório
plt.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    label="Classificador aleatório"
)

plt.title(
    "Curvas ROC dos modelos de classificação"
)

plt.xlabel(
    "Taxa de Falsos Positivos"
)

plt.ylabel(
    "Taxa de Verdadeiros Positivos"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    "graficos/curva_roc.png"
)

plt.close()


# ============================================================
# 23. CLUSTERIZAÇÃO — PREPARAÇÃO DOS DADOS
# ============================================================
scaler_cluster = StandardScaler()

X_cluster = scaler_cluster.fit_transform(
    X
)

print("\n================================")
print("     DADOS PARA CLUSTERIZAÇÃO")
print("================================")
print(
    "Formato dos dados:",
    X_cluster.shape
)


# ============================================================
# 24. CLUSTERIZAÇÃO — K-MEANS
# ============================================================
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    X_cluster
)

print("\n================================")
print("       RESULTADO K-MEANS")
print("================================")

print(
    "Quantidade de clusters:",
    3
)

print(
    "Clusters encontrados:",
    np.unique(clusters)
)


# ============================================================
# 25. QUANTIDADE DE AMOSTRAS POR CLUSTER
# ============================================================
quantidade_clusters = (
    pd.Series(clusters)
    .value_counts()
    .sort_index()
)

print("\nQuantidade de amostras por cluster:")
print(quantidade_clusters)

# Gráfico da quantidade de amostras
plt.figure(figsize=(8, 6))

quantidade_clusters.plot(
    kind="bar"
)

plt.title(
    "Quantidade de amostras por cluster"
)

plt.xlabel("Cluster")
plt.ylabel("Quantidade de amostras")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "graficos/tamanho_clusters.png"
)

plt.close()


# ============================================================
# 26. COMPARAÇÃO ENTRE CLASSES E CLUSTERS
# ============================================================
tabela_clusters = pd.crosstab(
    dados["classe"],
    clusters
)

print("\n================================")
print(" COMPARAÇÃO CLASSES X CLUSTERS")
print("================================")
print(tabela_clusters)


# ============================================================
# 27. SILHOUETTE SCORE
# ============================================================
silhouette = silhouette_score(
    X_cluster,
    clusters
)

print("\n================================")
print("     SILHOUETTE SCORE")
print("================================")

print(
    "Silhouette Score:",
    silhouette
)


# ============================================================
# 28. TESTE DE DIFERENTES VALORES DE K
# ============================================================
resultados_silhouette = []
valores_k = range(2, 7)

for k in valores_k:
    modelo_kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters_k = modelo_kmeans.fit_predict(
        X_cluster
    )

    score = silhouette_score(
        X_cluster,
        clusters_k
    )

    resultados_silhouette.append(
        score
    )

    print(
        f"K = {k} | Silhouette Score = {score:.4f}"
    )


# ============================================================
# 29. GRÁFICO DO SILHOUETTE SCORE
# ============================================================
plt.figure(figsize=(10, 6))

plt.plot(
    list(valores_k),
    resultados_silhouette,
    marker="o"
)

plt.title(
    "Silhouette Score para diferentes valores de K"
)

plt.xlabel(
    "Quantidade de clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.xticks(
    list(valores_k)
)

plt.grid()
plt.tight_layout()

plt.savefig(
    "graficos/silhouette_k.png"
)

plt.close()


# ============================================================
# 30. PCA PARA VISUALIZAÇÃO DOS CLUSTERS
# ============================================================
pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_cluster
)

print("\n================================")
print("             PCA")
print("================================")

print(
    "Formato original:",
    X_cluster.shape
)

print(
    "Formato após PCA:",
    X_pca.shape
)

print(
    "Variância explicada:"
)

print(
    pca.explained_variance_ratio_
)

print(
    "Variância explicada acumulada:",
    pca.explained_variance_ratio_.sum()
)


# ============================================================
# 31. GRÁFICO DOS CLUSTERS
# ============================================================
plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

plt.title(
    "Clusters encontrados pelo K-Means após redução com PCA"
)

plt.xlabel(
    "Componente Principal 1"
)

plt.ylabel(
    "Componente Principal 2"
)

plt.colorbar(
    label="Cluster"
)

plt.tight_layout()

plt.savefig(
    "graficos/clusters.png"
)

plt.close()


# ============================================================
# 32. GRÁFICO DAS CLASSES REAIS APÓS PCA
# ============================================================
plt.figure(figsize=(10, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=dados["classe"],
    cmap="viridis"
)

plt.title(
    "Distribuição das classes reais após PCA"
)

plt.xlabel(
    "Componente Principal 1"
)

plt.ylabel(
    "Componente Principal 2"
)

plt.colorbar(
    label="Classe real"
)

plt.tight_layout()

plt.savefig(
    "graficos/classes_pca.png"
)

plt.close()


# ============================================================
# FINALIZAÇÃO
# ============================================================
print("\n================================")
print("       ANÁLISE CONCLUÍDA")
print("================================")
print("\nGráficos salvos na pasta:")
print("graficos/")
