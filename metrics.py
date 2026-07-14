import numpy as np



# Classification Metrics


def accuracy(y_true, y_pred):

    correct = np.sum(y_true == y_pred)

    total = len(y_true)

    return correct / total


def precision(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))

    fp = np.sum((y_true == 0) & (y_pred == 1))

    if (tp + fp) == 0:
        return 0

    return tp / (tp + fp)


def recall(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))

    fn = np.sum((y_true == 1) & (y_pred == 0))

    if (tp + fn) == 0:
        return 0

    return tp / (tp + fn)


def f1_score(y_true, y_pred):

    p = precision(y_true, y_pred)

    r = recall(y_true, y_pred)

    if (p + r) == 0:
        return 0

    return (2 * p * r) / (p + r)


def confusion_matrix(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))

    tn = np.sum((y_true == 0) & (y_pred == 0))

    fp = np.sum((y_true == 0) & (y_pred == 1))

    fn = np.sum((y_true == 1) & (y_pred == 0))

    matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    return matrix



# Regression Metrics


def mean_absolute_error(y_true, y_pred):

    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):

    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)

    return np.sqrt(mse)


def r2_score(y_true, y_pred):

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

    ss_residual = np.sum((y_true - y_pred) ** 2)

    return 1 - (ss_residual / ss_total)



# Print Classification Report


def classification_report(y_true, y_pred):

    print("\n========== Classification Report ==========")

    print(f"Accuracy  : {accuracy(y_true, y_pred):.4f}")
    print(f"Precision : {precision(y_true, y_pred):.4f}")
    print(f"Recall    : {recall(y_true, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred):.4f}")

    print("\nConfusion Matrix\n")

    print(confusion_matrix(y_true, y_pred))


# Print Regression Report


def regression_report(y_true, y_pred):

    print("\n========== Regression Report ==========")

    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE  : {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE : {root_mean_squared_error(y_true, y_pred):.4f}")
    print(f"R²   : {r2_score(y_true, y_pred):.4f}")