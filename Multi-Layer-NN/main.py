from nn import NN
import numpy as np

# Create a neural network with 2 input neurons, 3 hidden neurons, and 1 output neuron
def main() -> None:
    nn = NN([2, 3, 1])
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([[0, 1, 1, 0]])
    nn.train(x, y)

    # test the neural network
    for i in range(4):
        print(f"Input: {x[:, i]}")
        print(f"Output: {nn.forward(x[:, i].reshape(-1, 1))}")
        print(f"Expected: {y[0, i]}")
        print()

if __name__ == "__main__":
    main()