# Importar bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Gerar dados de exemplo
data = {
    'Tamanho': [50, 60, 80, 100, 150, 200, 250],
    'Preço': [100000, 120000, 150000, 200000, 350000, 400000, 500000]
}
df = pd.DataFrame(data)

# Separar variáveis independentes (X) e dependentes (y)
X = df[['Tamanho']]  # Característica (Tamanho)
y = df['Preço']      # Alvo (Preço)

# Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Quadrático Médio (MSE): {mse}")
print(f"Coeficiente de Determinação (R²): {r2}")

# Visualizar os dados e a linha de regressão
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, model.predict(X), color='red', label='Linha de Regressão')
plt.xlabel('Tamanho (m²)')
plt.ylabel('Preço (R$)')
plt.legend()
plt.show()
