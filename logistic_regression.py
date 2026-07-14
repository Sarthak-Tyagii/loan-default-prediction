import numpy as np



# Sigmoid Function


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / ((1 + np.exp(-z)))



# Initialize Parameters


def initialize_parameters(n_features):

    weights = np.zeros((n_features, 1))
    bias = 0

    return weights, bias



# Forward Propagation


def forward(X, weights, bias):

    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)

    return predictions


# Binary Cross Entropy Cost


def compute_cost(y, predictions, class_weights=None):

    m = len(y)

    epsilon = 1e-10

    if class_weights is None:
        sample_weight = 1.0
    else:
        # class_weights = (weight_for_0, weight_for_1)
        sample_weight = np.where(y == 1, class_weights[1], class_weights[0])

    cost = (-1 / m) * np.sum(
        sample_weight * (
            y * np.log(predictions + epsilon)
            +
            (1 - y) * np.log(1 - predictions + epsilon)
        )
    )

    return cost


# Compute Gradients


def compute_gradients(X, y, predictions, class_weights=None):

    m = len(y)

    if class_weights is None:
        sample_weight = 1.0
    else:
        sample_weight = np.where(y == 1, class_weights[1], class_weights[0])

    error = sample_weight * (predictions - y)

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
    iterations,
    class_weights=None
):

    cost_history = []

    for i in range(iterations):

        predictions = forward(X, weights, bias)

        cost = compute_cost(y, predictions, class_weights)

        dw, db = compute_gradients(
            X,
            y,
            predictions,
            class_weights
        )

        weights = weights - learning_rate * dw

        bias = bias - learning_rate * db

        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i} | Cost = {cost:.5f}")

    return weights, bias, cost_history



# Train Model


def train_model(
    X_train,
    y_train,
    learning_rate=1.92,
    iterations=2000,
    balanced=True
):

    n_features = X_train.shape[1]

    weights, bias = initialize_parameters(n_features)

    class_weights = None
    if balanced:
        # The loan dataset is ~75% non-default / 25% default. Without
        # weighting, the cost function is minimized by mostly predicting
        # the majority class, which produces near-zero precision/recall
        # on defaults (the class we actually care about catching).
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        m = n_pos + n_neg
        weight_0 = m / (2 * n_neg)
        weight_1 = m / (2 * n_pos)
        class_weights = (weight_0, weight_1)
        print(f"Using balanced class weights -> Non-Default: {weight_0:.3f}, Default: {weight_1:.3f}")

    weights, bias, cost_history = gradient_descent(
        X_train,
        y_train,
        weights,
        bias,
        learning_rate,
        iterations,
        class_weights
    )

    return weights, bias, cost_history


# Predict Probability


def predict_probability(X,weights,bias):

    return forward(X,weights, bias )


# Predict Class


def predict(
    X,
    weights,
    bias,
    threshold=0.5
):

    probabilities = predict_probability(
        X,
        weights,
        bias
    )

    predictions = (probabilities >= threshold).astype(int)

    return predictions