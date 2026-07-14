import numpy as np


# Initialize Parameters

def initialize_parameters(n_features):

    weights = np.zeros((n_features, 1))
    bias = 0

    return weights, bias



# Forward Propagation


def forward(X, weights, bias):

    predictions = np.dot(X, weights) + bias

    return predictions



# Mean Squared Error Cost


def compute_cost(y, predictions):

    m = len(y)

    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)

    return cost



# Compute Gradients


def compute_gradients(X, y, predictions):

    m = len(y)

    error = predictions - y

    dw = (1 / m) * np.dot(X.T, error)

    db = (1 / m) * np.sum(error)

    return dw, db



# Gradient Descent


def gradient_descent(
    X,
    y,
    weights,
    bias,
    learning_rate,
    iterations
):

    cost_history = []

    for i in range(iterations):

        predictions = forward(X, weights, bias)

        cost = compute_cost(y, predictions)

        dw, db = compute_gradients(X, y, predictions)

        weights = weights - learning_rate * dw

        bias = bias - learning_rate * db

        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i} | Cost = {cost:.18f}")

    return weights, bias, cost_history



# Train Model

def train_model(
    X_train,
    y_train,
    learning_rate=1.92,
    iterations=2000
):

    n_features = X_train.shape[1]

    weights, bias = initialize_parameters(n_features)

    weights, bias, cost_history = gradient_descent(
        X_train,
        y_train,
        weights,
        bias,
        learning_rate,
        iterations
    )

    return weights, bias, cost_history



# Predict


def predict(
    X,
    weights,
    bias
):

    predictions = forward(
        X,
        weights,
        bias
    )

    return predictions