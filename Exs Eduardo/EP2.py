import pandas as pd
#from sklearn.preprocessing import StandardScaler  # não precisa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregue seu conjunto de dados CSV
data = pd.read_csv('class02.csv')

# Separe os recursos (99 primeiras colunas) e os rótulos (última coluna)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalização dos dados (opcional, mas pode ser útil) # SEM ISSO MELHOROU A PRECISÃO DO ALGORITMO
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

# Crie um classificador KNN com k = 10
k = 10
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Defina os índices para a validação cruzada com base nas primeiras 20% das linhas em cada fold
n_splits = 5
fold_size = len(X) // n_splits

# Realize a validação cruzada manualmente
scores = []

for i in range(n_splits):
    start = i * fold_size
    end = (i + 1) * fold_size if i < n_splits - 1 else len(X)
    
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
