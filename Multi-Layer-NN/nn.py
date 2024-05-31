import numpy as np

class NN:
    def __init__(self, layers: list[int]) -> None:
        self.layers_ = layers

        # Initialize weights and biases
        self.weights_ = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases_ = [np.random.randn(y, 1) for y in layers[1:]]
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        for w, b in zip(self.weights_, self.biases_):
            x = np.dot(w, x) + b
        return x

    def backward(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return sigmoid(self.forward(x)) - y
    
    def loss_derivative(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x - y
    
    def gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights_]
        dB = [np.zeros_like(b) for b in self.biases_]
        
        # Forward pass
        activations = [x]
        zs = []
        a = x
        for w, b in zip(self.weights_, self.biases_):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        
        # Backward pass
        delta = self.loss_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        dW[-1] = np.dot(delta, activations[-2].T)
        dB[-1] = delta
        
        for l in range(2, len(self.layers_)):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights_[-l+1].T, delta) * sp
            dW[-l] = np.dot(delta, activations[-l-1].T)
            dB[-l] = delta
        
        return dW, dB
    
    def train(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.1, epochs: int = 10_000) -> None:
        for _ in range(epochs):
            dW, dB = self.gradient(x, y)
            self.weights_ = [w - learning_rate * dw for w, dw in zip(self.weights_, dW)]
            self.biases_ = [b - learning_rate * db for b, db in zip(self.biases_, dB)]


# Activation functions

def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-Z))

def sigmoid_prime(Z: np.ndarray) -> np.ndarray:
    A = sigmoid(Z)
    return A * (1 - A)