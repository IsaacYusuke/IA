import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Dados do arquivo CSV
data = pd.read_csv('class01.csv')

# Divisao dos dados em conjuntos
# Conjunto de treinamento - 350 primeiras linhas
X_train = data.iloc[:350, :-1]
y_train = data.iloc[:350, -1]

# Conjunto de validacao - demais linhas
X_valid = data.iloc[350:, :-1]
y_valid = data.iloc[350:, -1]

# Pre-processamento de dados - Neste caso, nao interfere no resultado
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_valid = scaler.transform(X_valid)

# Inicializacao e treino do classificador Naive Bayes Gaussiano
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Precisao no conjunto de treinamento e validacao
y_train_pred = nb_classifier.predict(X_train)
y_valid_pred = nb_classifier.predict(X_valid)

# Acuracia nos conjuntos de treinamento e validacao
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_valid = accuracy_score(y_valid, y_valid_pred)

print("Acuracia no treinamento:", accuracy_train)
print("Acuracia na validacao:", round(accuracy_valid,4))