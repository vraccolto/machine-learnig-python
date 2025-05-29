from packaging import version
import sklearn
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
from sklearn.impute import SimpleImputer

# Carrega o arquivo .csv da pasta "data" em um DataFrame utilizando a biblioteca pandas, armazenando-o na variável "df".
df_vinhos = pd.read_csv(Path("data/group_6_winequality.csv"))

# Exibe em tela os dados das 5 primeiras linhas do df_vinhos.
print(df_vinhos.head())

# Exibe informações gerais sobre o df_vinhos, incluindo: número de linhas e colunas, nome de cada coluna, quantidade de valores não nulos e tipo de dado de cada coluna.
print(df_vinhos.info())


# value_counts() verifica a frequência de linhas idênticas dentro do df_vinhos.
# Podemos perceber que o df_vinhos tem 6497 linhas. Após aplicar df_vinhos.value_counts(), restaram 5209 linhas únicas. Isso indica que existem 1288 linhas duplicadas.
print(df_vinhos.value_counts())

# DADOS FALTANTES

# Certifique-se de que a coluna 'alcohol' esteja numérica
df_vinhos['alcohol'] = pd.to_numeric(df_vinhos['alcohol'], errors='coerce')

# Selecionar apenas colunas numéricas
colunas_numericas = df_vinhos.select_dtypes(include=['float64', 'int64']).columns

# Cria o imputador
imputador = SimpleImputer(strategy="median")

# Aplica o imputador às colunas numéricas
df_vinhos[colunas_numericas] = imputador.fit_transform(df_vinhos[colunas_numericas])

# Imputador para colunas categóricas
imputador_cat = SimpleImputer(strategy="most_frequent")

# Aplica na coluna 'type'
df_vinhos[['type']] = imputador_cat.fit_transform(df_vinhos[['type']])

"""# Análise exploratória:

## Histogramas
"""

# Vamos importar o matplotlib para termos uma visualização gráfica de cada
# uma das nossas features. Vamos fazer um histograma para cada classe.
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

df_vinhos.hist(bins=50, figsize=(20, 10))
plt.show()

"""## scatter plot"""

df_vinhos.plot( x="citric acid", y="pH", grid=True)
plt.show()

# Temos no dataset. Um scatter plot
df_vinhos.plot(kind="scatter", x="quality", y="free sulfur dioxide", grid=True)
plt.show()

# Podemos melhorar ainda mais.
df_vinhos.plot(kind="scatter", x="quality", y="free sulfur dioxide", grid=True, alpha=0.5)
plt.show()

# Vamos adicionar uma cor para o preço das casas
df_vinhos.plot(kind="scatter", x="quality", y="free sulfur dioxide", grid=True,
s=df_vinhos["chlorides"] / 100, label="chlorides",
c="residual sugar", cmap="jet", colorbar=True,
legend=True, sharex=False, figsize=(10, 7))
plt.show()

"""## Correlação dos dados"""

# Vamos procurar correlações nos nossos dados. Como temos dados com texto, iremos
# passar o argumento numeric_only para ignorar o que não for número.
corr_matrix = df_vinhos.corr(numeric_only = True)

# Vamos ver quem tem a maior correlação com o valor das casas
corr_matrix["quality"].sort_values(ascending=False)

print(corr_matrix)

# Deu para ver que não ajudou muito, logo, vamos fazer um heatmap de correlação
# utilizando o seaborn
import seaborn as sns

sns.heatmap(corr_matrix, cmap="crest", annot=True, fmt=".1f")
plt.show()

"""# Limpeza e pré-processamento:"""

from sklearn.preprocessing import StandardScaler

# Separando atributos e alvo
# Drop the non-numeric column 'type' before scaling
X = df_vinhos.drop(["quality", "type"], axis=1)
y = df_vinhos["quality"]

# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# verificar se exixti dados faltantes

df_vinhos.isnull().sum()

from sklearn.ensemble import IsolationForest

# Seleciona apenas colunas numéricas
X = df_vinhos.select_dtypes(include=[np.number])

# Treina o modelo
iso = IsolationForest(contamination=0.01, random_state=42)
outlier_pred = iso.fit_predict(X)

# Marcar -1 como outliers
df_vinhos['outlier'] = outlier_pred == -1

# Ver quantos outliers foram detectados
df_vinhos['outlier'].value_counts()

"""# Train / Tests Split"""

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df_vinhos, test_size=0.2, random_state=42)

print("Tamanho dos dados de treino: ", train_set.shape[0])
print("Tamanho dos dados de teste: ", len(test_set))

"""# Modelagem"""

X = df_vinhos.drop(columns=["quality","type","outlier"],errors="ignore")
Y = df_vinhos["quality"]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""## Ajuste de hiperparamentro"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Pipeline completo: escalonamento, transformação polinomial, regressão
pipeline_poly = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(include_bias=False)),
    ("lr", LinearRegression())
])

# Espaço de busca: testar graus de 1 a 4
param_grid = {
    "poly__degree": [1, 2, 3, 4]
}

# Busca com validação cruzada
grid_poly = GridSearchCV(pipeline_poly,
                         param_grid,
                         cv=5,
                         scoring="r2",
                         n_jobs=-1)

grid_poly.fit(X_train, y_train)

# Predição no conjunto de teste
y_pred_poly = grid_poly.predict(X_test)

# Resultados
print("Melhores parâmetros:", grid_poly.best_params_)
print("Melhor score de validação (R²):", grid_poly.best_score_)
print("Score no teste (R²):", r2_score(y_test, y_pred_poly))

"""## MODELO 1: Regressão Linear"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cria modelo com padronização dos dados
modelo_linear = make_pipeline(StandardScaler(), LinearRegression())

# Treina o modelo
modelo_linear.fit(X_train, y_train)

# Faz previsões
y_pred_linear = modelo_linear.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("=== Regressão Linear ===")
print(f"MSE: {mse_linear:.2f}")
print(f"R² : {r2_linear:.2f}\n")

"""## MODELO 2: Regressão Polinomial"""

modelo_poli = make_pipeline(StandardScaler(), PolynomialFeatures(degree=1), LinearRegression())
modelo_poli.fit(X_train, y_train)
y_pred_poli = modelo_poli.predict(X_test)
mse_poli = mean_squared_error(y_test, y_pred_poli)
r2_poli = r2_score(y_test, y_pred_poli)

print("=== Regressão Polinomial (grau 2) ===")
print(f"MSE: {mse_poli:.2f}")
print(f"R² : {r2_poli:.2f}\n")

"""## MODELO 3: SVR"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Importar SVR do módulo sklearn.svm
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

modelo_svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', ))
modelo_svr.fit(X_train, y_train)
y_pred_svr = modelo_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print("=== SVR (RBF Kernel) ===")
print(f"MSE: {mse_svr:.2f}")
print(f"R² : {r2_svr:.2f}")

import matplotlib.pyplot as plt

def plot_resultados(y_test, y_preds, nomes_modelos):
    plt.figure(figsize=(18, 5))

    for i, (y_pred, nome) in enumerate(zip(y_preds, nomes_modelos)):
        plt.subplot(1, 3, i+1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Qualidade Real")
        plt.ylabel("Qualidade Prevista")
        plt.title(nome)
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# Após treinar os modelos e gerar y_pred_linear, y_pred_poli, y_pred_svr:
plot_resultados(
    y_test,
    [y_pred_linear, y_pred_poli, y_pred_svr],
    ["Regressão Linear", "Regressão Polinomial", "SVR (RBF)"]
)

"""# Classificação

## Requisitos:
"""

# Recarregar o DataFrame original, caso tenha sido sobrescrito
# dataset = pd.read_csv(Path("data/group_6_winequality.csv"))

# Criar nova variável alvo categórica com base em faixas da coluna 'quality'
# Exemplo 1: Faixas simples
def criar_categoria_qualidade_faixa1(qualidade):
    if 0 <= qualidade <= 3:
        return 'Baixa Qualidade'
    elif 4 <= qualidade <= 6:
        return 'Qualidade Média'
    else:  # 7 <= qualidade <= 10
        return 'Alta Qualidade'

# Aplicar a função à coluna 'quality' do DataFrame
df_vinhos['categoria_qualidade_faixa1'] = df_vinhos['quality'].apply(criar_categoria_qualidade_faixa1)

print("Contagem de valores para Faixa 1:")
print(df_vinhos['categoria_qualidade_faixa1'].value_counts())
print("-" * 30)

# Exemplo 2: Testando faixas diferentes
def criar_categoria_qualidade_faixa2(qualidade):
    if qualidade <= 5:
        return 'Abaixo da Média'
    elif qualidade == 6:
        return 'Na Média'
    else:  # qualidade >= 7
        return 'Acima da Média'

# Aplicar a função à coluna 'quality'
df_vinhos['categoria_qualidade_faixa2'] = df_vinhos['quality'].apply(criar_categoria_qualidade_faixa2)

print("Contagem de valores para Faixa 2:")
print(df_vinhos['categoria_qualidade_faixa2'].value_counts())
print("-" * 30)

# Exemplo 3: Testando mais faixas
def criar_categoria_qualidade_faixa3(qualidade):
    if qualidade < 5:
        return 'Ruim'
    elif qualidade == 5:
        return 'Regular'
    elif qualidade == 6:
        return 'Boa'
    elif qualidade > 6:
        return 'Excelente'

# Aplicar a função à coluna 'quality'
df_vinhos['categoria_qualidade_faixa3'] = df_vinhos['quality'].apply(criar_categoria_qualidade_faixa3)

print("Contagem de valores para Faixa 3:")
print(df_vinhos['categoria_qualidade_faixa3'].value_counts())
print("-" * 30)

import seaborn as sns
import matplotlib.pyplot as plt

# Criar uma figura com 3 subplots lado a lado
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Gráfico 1 - Faixa 1
sns.countplot(x='categoria_qualidade_faixa1', data=df_vinhos, ax=axes[0])
axes[0].set_title('Faixa 1')
axes[0].set_xlabel('Categoria')
axes[0].set_ylabel('Contagem')

# Gráfico 2 - Faixa 2
sns.countplot(x='categoria_qualidade_faixa2', data=df_vinhos, ax=axes[1])
axes[1].set_title('Faixa 2')
axes[1].set_xlabel('Categoria')
axes[1].set_ylabel('')

# Gráfico 3 - Faixa 3
sns.countplot(x='categoria_qualidade_faixa3', data=df_vinhos, ax=axes[2])
axes[2].set_title('Faixa 3')
axes[2].set_xlabel('Categoria')
axes[2].set_ylabel('')

# Ajustar layout para evitar sobreposição
plt.tight_layout()
plt.show()

"""# Preparação

"""