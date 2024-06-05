import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def sigmoid(z):
    X = np.exp(z)
    return X / (1 + X)


def sigmoid_der(z):
    A = sigmoid(z)
    return A * (1 - A)


def initialize_parameters(n_x, n_h, n_y):
    return {
        "W1": np.random.randn(n_h, n_x) * 0.01,
        "b1": np.zeros([n_h, 1]),
        "W2": np.random.randn(n_y, n_h) * 0.01,
        "b2": np.zeros([n_y, 1]),
    }


def forward_propagation(X, parameters):
    # Hidden Layer
    Z1 = parameters["W1"].dot(X) + parameters["b1"]
    A1 = np.tanh(Z1)  # change here for the targil
    # Output Layer
    Z2 = parameters["W2"].dot(A1) + parameters["b2"]
    A2 = sigmoid(Z2)
    # print(A2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def tanh_der(z):
    X = np.tanh(z)
    return 1 - X**2


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]  # Number of samples
    # Assuming the Log_loss function is used -- like last time:
    dZ2 = cache["A2"] - Y  # for the sigmoid layer
    dW2 = (1 / m) * dZ2.dot(cache["A1"].T)
    db2 = (1 / m) * np.sum(dZ2)
    # Output Layer--using MSE cost
    # dA2 =  -1 * (Y- cache["A2"]) #The derivative of MSE is -(Y-YP) (derivative of cost)-- this is in slide 54
    # dZ2 = dA2 * sigmoid_der(cache["Z2"]) #node derivative * input derivative
    # dW2 = (1 / m) * np.dot(dZ2,cache["A1"].T ) #for the input- A1 is the input to the second level, as X is the input to the first level
    # db2 = (1 / m) * np.sum(dZ2)
    # Hidden Layer
    dA1 = np.dot(parameters["W2"].T, dZ2)
    dZ1 = dA1 * tanh_der(cache["Z1"])  # change here for the targil
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1)
    # db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2}


def LogLoss_calculation(A, Y):
    cost = np.mean(-(Y * np.log(A) + (1 - Y) * np.log(1 - (A))))
    return cost


def update_parameters(parameters, grads, learning_rate):
    return {
        "W1": parameters["W1"] - learning_rate * grads["dW1"],
        "W2": parameters["W2"] - learning_rate * grads["dW2"],
        "b1": parameters["b1"] - learning_rate * grads["db1"],
        "b2": parameters["b2"] - learning_rate * grads["db2"],
    }


def nn_model(X, Y, iterations, lr):
    n_x = X.shape[0]
    n_h = 5  # change here for the targil
    n_y = 1
    parameters = initialize_parameters(n_x, n_h, n_y)
    print("Network shape ", X.shape[0], n_h, n_y)
    for i in range(iterations):
        A2, cache = forward_propagation(X, parameters)
        # cost = MSE_calculation(A2,Y)
        cost = LogLoss_calculation(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, lr)
        costs.append(cost)
        # cost check
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    return parameters, costs


def predict(X, parameters):
    A2, _ = forward_propagation(X, parameters)
    return np.rint(A2)


def prediction_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def main() -> None:
    url = "https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true"
    # url = 'https://github.com/rosenfa/nn/blob/master/class2/spam.csv?raw=true'
    df = pd.read_csv(url, header=0)
    features = df.drop(["Outcome"], axis=1)
    features = (features - features.mean()) / features.std()
    X = np.array(features)
    Y = np.array(df["Outcome"])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
    df

    from sklearn.linear_model import LogisticRegression

    sk_model = LogisticRegression()
    sk_model.fit(X_train, Y_train)
    accuracy = sk_model.score(X_test, Y_test)
    print("accuracy = ", accuracy * 100, "%")

    X_train, X_test = X_train.T, X_test.T

    num_iterations = 2000  # number of iterations  #change here for the targil
    alpha = 1  # learning rate
    costs = []
    parameters, costs = nn_model(X_train, Y_train, num_iterations, alpha)
    Y_train_predict = predict(X_train, parameters)
    train_acc = prediction_accuracy(Y_train_predict, Y_train)
    Y_test_predict = predict(X_test, parameters)
    test_acc = prediction_accuracy(Y_test_predict, Y_test)
    parameters["train_accuracy"] = train_acc
    parameters["test_accuracy"] = test_acc

    plt.plot(costs)

    print("Training acc : ", str(train_acc))
    print("Testing acc : ", str(test_acc))


if __name__ == "__main__":
    main()
