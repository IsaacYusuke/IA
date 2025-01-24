# Logistic Regression - usa função logística (sigmóide) = 1/(1+exp(-z)) para classificação binária
# z = w1*x1 + w2*x2 + ... + b

# Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Simular dataset
np.random.seed(42)
n_samples = 200
idade = np.random.randint(18, 80, size=n_samples)
colesterol = np.random.randint(150, 300, size=n_samples)
# Gerar rótulos (1: possui condição médica, 0: não possui)
condicao = (0.3 * idade + 0.3 * colesterol + np.random.normal(0, 15, n_samples)) > 100
condicao = condicao.astype(int)

# Criar DataFrame
data = pd.DataFrame({'Idade': idade, 'Colesterol': colesterol, 'Condicao': condicao})

# Separar variáveis independentes (X) e dependente (y)
X = data[['Idade', 'Colesterol']]
y = data['Condicao']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar as variáveis (opcional, mas recomendado)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar o modelo de Regressão Logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
plt.title("Matriz de Confusão")
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xticks([0, 1], ["Não Condição", "Condição"])
plt.yticks([0, 1], ["Não Condição", "Condição"])
plt.xlabel("Predição")
plt.ylabel("Verdadeiro")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.show()

# Visualizar os dados e a decisão
plt.figure(figsize=(8, 6))
plt.scatter(data['Idade'], data['Colesterol'], c=data['Condicao'], cmap='bwr', alpha=0.7, label="Dados Reais")
plt.title("Classificação por Idade e Colesterol")
plt.xlabel("Idade")
plt.ylabel("Colesterol")
plt.legend()
plt.show()
