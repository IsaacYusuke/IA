# PCS3438 - Inteligência Artificial - 2023/2
# Template para aula de laboratório em Redes Neurais - 20/09/2023

#Nome 1: Isaac Yusuke Yanagui
#NUSP 1: 10772369

#Nome 2: Guilherme Rodrigues Bastos
#NUSP 2: 10416851

"""
3) Obter o dataset Breast Cancer Winsconsin Dataset (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) (3.0)
    3.1) Realizar a divisão do dataset em subset de treino (80%) e de teste (20%)
    3.2) Realizar a normalização das features (Z-Score ou Max-min)
    3.3) Realizar o treinamento do MLP desenvolvido para classificação do dataset de teste (já com a regularização)
    3.4) Reportar a acurácia da classificação
    3.5) Reportar a matriz de confusão da classificação
"""

import numpy as np
from sklearn.datasets import load_breast_cancer #versao 3.0


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray):
    return x * (1 - x)


def mse_loss(y: np.ndarray, y_hat: np.ndarray):
    return np.mean(np.power(y - y_hat, 2))


def mse_loss_derivative(y: np.ndarray, y_hat: np.ndarray):
    return y_hat - y


class Layer:
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = 2 * np.random.random((input_dim, output_dim)) - 1
        self.biases = np.zeros((1, output_dim))
        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None

    def forward(self, input_data) -> np.ndarray:
        """
        TODO: Forward pass of a single MLP layer

        Args:
            input_data (np.ndarray): Input data

        Returns:
            np.ndarray: Output of the layer
        """

        #raise NotImplementedError

        self.input = input_data
        self.output = sigmoid(input_data.dot(self.weights) + self.biases)
        return self.output

    def backward(self, output_error: np.ndarray, learning_rate: float, lambda_reg: float) -> np.ndarray:
        """
        TODO: Implement backward pass of a single MLP layer

        This method calculates the error of the layer and updates the weights and biases

        Args:
            output_error (np.ndarray): Error of the output layer
            learning_rate (float): Learning rate

        Returns:
            np.ndarray: Error of the previous layer
        """

        #raise NotImplementedError
        
        gradient = output_error * sigmoid_derivative(self.output)
        input_error = gradient.dot(self.weights.T)
        
        self.weights += self.input.T.dot(gradient) * learning_rate - lambda_reg * self.weights  #versao 2.0
        
        return input_error


def forward(input: np.ndarray, layers: list[Layer]):
    """
    TODO: Applies forward pass to the MLP model

    Args:
        input (np.ndarray): Input data
        layers (list[Layer]): List of layers

    Returns:
        np.ndarray: Output of the MLP model
    """
    #raise NotImplementedError
    
    data = input
    for layer in layers:
        data = layer.forward(data)
    return data




def backward(
    y: np.ndarray, y_hat: np.ndarray, layers: list[Layer], learning_rate: float, lambda_reg: float
) -> None:
    """
    TODO: Applies backpropagation to the MLP model

    Args:
        y (np.ndarray): Ground truth
        y_hat (np.ndarray): Predicted values
        layers (list[Layer]): List of layers
        learning_rate (float): Learning rate
    """

    #raise NotImplementedError
    
    error = y - y_hat
    for i in range(len(layers)):
        error = layers[-i -1].backward(error, learning_rate, lambda_reg)


def main():
    # versao 3.0 - download da base de dados
    # Classes: 2
    # Samples per class :212(M),357(B)
    # Samples total: 569
    # Dimensionality: 30
    # Features: real, positive
    data = load_breast_cancer()
    x_full = data.data
    y_full = np.array([data.target]).reshape(-1, 1) #versao 3.0 - transforma vetor em matriz? np.array([data.target])
    
    # versao 3.0 - 3.2) Realizar a normalização das features (Z-Score ou Max-min) - usa Max-min
    x_full = (x_full - np.min(x_full))/(np.max(x_full) - np.min(x_full))
    
    # versao 3.0 - 3.1) Realizar a divisão do dataset em subset de treino (80%) e de teste (20%)
    train_indices = np.random.permutation(len(x_full))
    cut_index = int(0.8*len(x_full))
    x = x_full[train_indices][0:cut_index]
    y = y_full[train_indices][0:cut_index]
    x_test = x_full[train_indices][cut_index:]
    y_test = y_full[train_indices][cut_index:]

    # Hyperparameters
    batch_size = 200  # Tamanho do mini lote - versao 3.0 - aumenta o tamanho
    hidden_layers = [2, 2]  # Two hidden layers, 2 neurons each
    epochs = 100000
    learning_rate0 = 0.5   #0.1  #versao 1.0 - nao converge se a lr inicial for mt baixa? - convergencia aumentou com lr maior
    learning_rate = learning_rate0
    lambda_reg = 0.0005   #versao 2.0  - termo de regularização L2 - não converge mais? - convergiu com valores bem baixos de lambda_reg

    # Initialize layers
    layers = [Layer(x.shape[1], hidden_layers[0])]
    for i in range(len(hidden_layers) - 1):
        layers.append(Layer(hidden_layers[i], hidden_layers[i + 1]))
    layers.append(Layer(hidden_layers[-1], y.shape[1]))
    
    i = 2  #versao 1.0

    # Train the model
    for epoch in range(epochs):
        # versao 1.0 - Embaralhar os dados
        indices = np.random.permutation(len(x))
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for i in range(0, len(x), batch_size): # versao 1.0 - Treinamento nos mini lotes (batches)
            x_batch = x_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Forward pass e backward para o mini lote
            y_hat = forward(x_batch, layers)
            loss = mse_loss(y_batch, y_hat)
            backward(y_batch, y_hat, layers, learning_rate, lambda_reg)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} Loss: {np.mean(loss)}")
        
        learning_rate = learning_rate0/i   # versao 1.0 - pq nao converge? - versao 3.0 - convergencia melhorou sem essa linha
        i = i + 1                         # versao 1.0 - R: melhorou quando colocou os batches tambem!

    # Test the model
    y_hat = forward(x_test, layers)

    print("Test input:", x_test)
    print("Test error:", y_hat - y_test)


if __name__ == "__main__":
    main()
