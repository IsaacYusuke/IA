import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np

# Carregue os dados do arquivo CSV
data = pd.read_csv('reg01.csv')

# Separacao dos recursos (X) e o alvo (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Inicializacao das listas para armazenar os RMSEs de treinamento e validacao
rmse_train_list = []
rmse_test_list = []

# Instancia do modelo Lasso com alpha=1
lasso = Lasso(alpha=1)

# Validacao cruzada Leave-One-Out
for i in range(len(X)):
    X_train = np.delete(X.values, i, axis=0)
    y_train = np.delete(y.values, i, axis=0)
    X_test = X.values[i:i+1]
    y_test = y.values[i:i+1]

    lasso.fit(X_train, y_train)
    
    # Calcule o RMSE de treinamento
    y_train_pred = lasso.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_train_list.append(rmse_train)

    # Calcule o RMSE de validacao
    y_pred = lasso.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_test_list.append(rmse_test)

# Calcule o valor medio do RMSE para treinamento e validacao
average_rmse_train = np.mean(rmse_train_list)
average_rmse_test = np.mean(rmse_test_list)

print("Valor RMSE de treinamento:", round(average_rmse_train,5))
print("Valor RMSE de validacao:", round(average_rmse_test,5))
