import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregando o conjunto de dados CSV
#data = pd.read_csv('class02.csv')  #As vezes nao funciona???
data = pd.read_csv(r'C:\Users\yusuk\Documents\Poli\2023\IA\Exs Eduardo\class02.csv')
#data = pd.read_csv(r'C:\Users\yusuk\OneDrive\Tesouraria Águias de Haia\Poli\2023\IA\Exs Eduardo\class02.csv')  #Caminho completo

# Separando os atributos (99 primeiras colunas) e os rótulos (última coluna)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Criando um classificador KNN com k = 10
k = 10
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Definindo os índices para os 5 folds
n_splits = 5
fold_size = len(X) // n_splits

# Realizando a validação cruzada manualmente
scores = []

for i in range(n_splits):
    start = i * fold_size
    end = (i + 1) * fold_size
    
    train_indices = list(range(0, start)) + list(range(end, len(X)))
    test_indices = list(range(start, end))
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)

# Exiba a acurácia média e os resultados para cada fold
for fold, score in enumerate(scores, start=1):
    print(f'Acurácia do Fold {fold}: {score * 100:.2f}%')

print(f'Acurácia Média: {sum(scores) / n_splits * 100:.2f}%')
