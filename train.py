import os
import numpy as np
from model_io import save_model
from visualization import plot_cost

from preprocessing import preprocess

from logistic_regression import (
    train_model,
    predict
)

from metrics import (
    classification_report
)



# Load and Preprocess Dataset


# Relative path -> works regardless of who runs it / which machine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Loan_Default.csv")

X_train, X_test, y_train, y_test = preprocess(file_path)

print("\nData Preprocessing Completed")

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])



# Train Logistic Regression


print("\nTraining Logistic Regression...\n")

weights, bias, cost_history = train_model(
    X_train,
    y_train,
    learning_rate=1.92,
    iterations=2000
)

print("\nTraining Completed Successfully")


# Save Model


save_model(weights, bias)


# Predict


predictions = predict(
    X_test,
    weights,
    bias
)


# Evaluate Model


classification_report(
    y_test,
    predictions
)



# Display Final Parameters


print("\nModel Parameters")

print("Bias :", bias)

print("Weights Shape :", weights.shape)
print()

plot_cost(cost_history)
