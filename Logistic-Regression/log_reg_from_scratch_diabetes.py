import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

#TODO: convert the sigmoid function to use iteration instead of vectorized operations
def sigmoid_iterative(z: np.ndarray) -> np.ndarray:
    result = np.zeros(z.shape[0])
    for i in range(z.shape[0]):
        result[i] = 1 / (1 + np.exp(-z[i]))
    return result


def loss(h: np.ndarray, y: np.ndarray) -> float:
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

#TODO: convert the loss function to use iteration instead of vectorized operations
def loss_iterative(h: np.ndarray, y: np.ndarray) -> float:
    result = 0
    for i in range(h.shape[0]):
        result += (-y[i] * np.log(h[i]) - (1 - y[i]) * np.log(1 - h[i]))
    return result / h.shape[0]


def predict_probs(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return sigmoid(np.dot(X, theta))


def predict(X: np.ndarray, theta: np.ndarray, threshold: float = 0.5) -> int:
    if predict_probs(X, theta) >= threshold:
        return 1
    return 0


def load_data() -> pd.DataFrame:
    url = "https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true"
    df_pima = pd.read_csv(url, header=0)
    return df_pima


def split_data(df_pima: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df_pima.iloc[:, :-1].values  # everything except the target
    y = df_pima.iloc[:, -1].values  # the target

    return X, y


def standardize(X: np.ndarray) -> np.ndarray:
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.append(np.ones([len(X), 1]), X, 1)


def test_model(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    correct = 0
    for x, y in zip(X, y):
        p = predict(x, theta)
        if p == y:
            correct += 1

    # m = len(y)
    m = X.shape[0]
    accuracy = correct / m * 100
    return accuracy

def train_model(X_train: np.ndarray, y_train: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, list]:
    # fit the data
    cost_array = (
        []
    )  # keeping a list of the cost at each iteration to make sure it is constantly decreasing
    iterations = 2000  # like the red arrow in slide 37
    lr = 0.01
    m, n = X_train.shape
    for i in range(iterations):
        Z = np.dot(X_train, theta)
        A = sigmoid(Z)  # also called h for 'hypothesis'
        Gradient = np.dot(X_train.T, (A - y_train)) / m
        theta -= lr * Gradient
        cost = loss(A, y_train)
        cost_array.append(cost)

    return theta, cost_array

def dot_product(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # iterative version of the numpy dot product
    result = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            result[i] += X[i][j] * theta[j]
    return result
    

def train_model_iterative(X_train: np.ndarray, y_train: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, list]:
    """This function is doing exactly the same thing as the previous one, but it is using only built in python loops and not vectorized operations. This is to show the difference in speed between the two."""
    # fit the data
    cost_array = (
        []
    )  # keeping a list of the cost at each iteration to make sure it is constantly decreasing
    iterations = 2000  # like the red arrow in slide 37
    lr = 0.01
    m, n = X_train.shape
    for i in range(iterations):
        Z = dot_product(X_train, theta)
        A = sigmoid(Z)  # also called h for 'hypothesis'
        Gradient = np.zeros(n)
        for j in range(n):
            for k in range(m):
                Gradient[j] += X_train[k][j] * (A[k] - y_train[k]) / m
        
        # theta -= lr * Gradient
        for j in range(n):
            theta[j] -= lr * Gradient[j]
        cost = loss(A, y_train)
        cost_array.append(cost)

    return theta, cost_array

def main() -> None:
    np.random.seed(42)

    df_pima = load_data()
    X, y = split_data(df_pima)
    X = standardize(X)
    X = add_bias(X)

    theta = np.zeros(X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    theta, cost_array = train_model_iterative(X_train, y_train, theta)

    plt.plot(cost_array)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
    # plotting the cost against the number of iterations to make sure our model is improving the whole time

    accuracy = test_model(X_test, y_test, theta)
    print(f"accuracy: {accuracy:.2f}%")

    sk_model = LogisticRegression()
    sk_model.fit(X_train, y_train)

    accuracy = sk_model.score(X_test, y_test) * 100
    print(f"accuracy: {accuracy:.2f}%")

    print("Coefficients: \n", sk_model.coef_)
    print(theta)

if __name__ == "__main__":
    main()
