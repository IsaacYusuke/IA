# PCS3438 - Inteligência Artificial - 2023/2
# Template para aula de laboratório em Redes Neurais - 20/09/2023

#Nome 1: Isaac Yusuke Yanagui
#NUSP 1: 10772369

#Nome 2: Guilherme Rodrigues Bastos
#NUSP 2: 10416851

"""
    1) Construir uma rede neural do tipo MLP com número configurável de camadas escondidas que seja treinável pelo algoritmo SGD (3.0)
    A única diferença dessa entrega e o trabalho realizado no laboratório é o "S" do SGD, que envolve o treinamento utilizando:
    1.1) Grupos de exemplos a cada iteração, ao invés do dataset de treino inteiro. Esses grupos são tipicamente chamados de "batches".
    1.2) A learning rate inicial deve ser reduzida a cada passo para auxiliar a convergência. Formalmente ela deve ser quadraticamente somável.
    Para lr_0 < 1.0, a expressão de decrescimento lr_i = lr_{i-1}/(1+i) é suficiente.
"""

import numpy as np


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

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
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
        
        self.weights += self.input.T.dot(gradient) * learning_rate
        
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
    y: np.ndarray, y_hat: np.ndarray, layers: list[Layer], learning_rate: float
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
        error = layers[-i -1].backward(error, learning_rate)


def main():
    # XOR input and output
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Hyperparameters
    hidden_layers = [2, 2]  # Two hidden layers, 2 neurons each
    epochs = 100000
    learning_rate = 0.1

    # Initialize layers
    layers = [Layer(x.shape[1], hidden_layers[0])]
    for i in range(len(hidden_layers) - 1):
        layers.append(Layer(hidden_layers[i], hidden_layers[i + 1]))
    layers.append(Layer(hidden_layers[-1], y.shape[1]))

    # Train the model
    for epoch in range(epochs):
        # Forward pass
        y_hat = forward(x, layers)

        # Loss
        loss = mse_loss(y, y_hat)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} Loss: {np.mean(loss)}")

        # Backward
        backward(y, y_hat, layers, learning_rate)

    # Test the model
    y_hat = forward(x, layers)

    print("Test input:", x)
    print("Test output:", y_hat)


if __name__ == "__main__":
    main()
