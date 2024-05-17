import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=["Price"])
    return X, y


def preprocess_data(X: pd.DataFrame, y: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_scaled = preprocessing.scale(X)
    y_scaled = preprocessing.scale(y)
    return X_scaled, y_scaled


def split_data(
    X: pd.DataFrame, y: pd.DataFrame, split_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)
    return X_train, X_test, y_train, y_test


def train_model_built_in(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_model_manual(
    X_train: np.ndarray, y_train: np.ndarray, lambda_: float = 0.1
) -> np.ndarray:
    """
    Train the model manually using the Normal Equation.
    """
    # Add a small bias term to the diagonal elements of the matrix
    bias = lambda_ * np.eye(X_train.shape[1])
    theta = np.linalg.inv(X_train.T.dot(X_train) + bias).dot(X_train.T).dot(y_train)
    return theta


def error_function(y_actual, y_predicted):
    """
    MSE = mean squared error
    1/m * sum((y_actual - y_predicted)^2)
    """
    error = np.sum((y_actual - y_predicted) ** 2)
    return error / len(y_actual)


def gradient_descent_calc(y_actual, y_pred, x):
    grad = np.dot(x.T, (y_pred - y_actual))
    return grad / len(y_actual)


def gradient_descent(X_train, y_train, theta, alpha=0.01, iterations=1000):
    """
    theta = theta - alpha * (X_train.T.dot(X_train.dot(theta) - y_train))
    """
    theta = np.zeros((X_train.shape[1], 1))
    for _ in range(iterations):
        theta = theta - alpha * gradient_descent_calc(y_train, X_train.dot(theta), X_train)
    return theta


def predict(thetas: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Predicts the output values for a given set of input features using the trained model.

    Parameters:
    thetas (np.ndarray): The trained model parameters.
    X_test (np.ndarray): The input features for which to make predictions.

    Returns:
    np.ndarray: The predicted output values.
    """
    # thetas = thetas.reshape(-1, 1)
    return X_test.dot(thetas)


def evaluate_model(theta: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = predict(theta, X_test)
    return mean_squared_error(y_test, y_pred)


def main() -> None:
    # setting up the np.random seed for reproducibility
    np.random.seed(42)

    X, y = load_data()
    X_scaled, y_scaled = preprocess_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled, 0.2)

    # add another dimension to the X_train and X_test for the bias term
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    model = train_model_built_in(X_train, y_train)
    error = mean_squared_error(y_test, model.predict(X_test))
    print(f"Error: {error}")

    print(model.coef_)

    theta = train_model_manual(X_train, y_train)
    error = evaluate_model(theta, X_test, y_test)
    print(f"Error: {error}")

    print(theta)

    theta = gradient_descent(X_train, y_train)
    error = evaluate_model(theta, X_test, y_test)
    print(f"Error: {error}")

    print(theta)


if __name__ == "__main__":
    main()
