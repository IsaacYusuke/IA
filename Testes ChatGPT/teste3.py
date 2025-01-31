#Classificar imagens com rede neural - digitos manuscritos

# Importar bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Carregar o dataset MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Sortear 6 imagens aleatórias do conjunto de treino
indices_aleatorios = np.random.choice(len(X_train), 6, replace=False)

# Visualizar algumas imagens do dataset
plt.figure(figsize=(10, 5))
for i, idx in enumerate(indices_aleatorios):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.title(f"Label: {y_train[idx]}")
    plt.axis('off')
plt.show()

# Pré-processamento dos dados
# Normalizar os valores dos pixels para [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Transformar rótulos para formato categórico (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Construir o modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),   # Achatar imagens de 28x28 para 1D (28*28=784)
    Dense(128, activation='relu'),  # Camada densa com 128 neurônios
    Dense(10, activation='softmax') # Saída com 10 classes (0-9)
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Teste - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

# Visualizar algumas previsões
predictions = model.predict(X_test)

# Sortear 6 imagens aleatórias do conjunto de teste
indices_aleatorios = np.random.choice(len(X_test), 6, replace=False)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(indices_aleatorios):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f"Real: {np.argmax(y_test[idx])}, Pred: {np.argmax(predictions[idx])}")
    plt.axis('off')
plt.show()
