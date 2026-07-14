import os
import json
import numpy as np
import pandas as pd

from preprocessing import preprocess, preprocess_customer

from logistic_regression import (
    train_model,
    predict,
    predict_probability
)

from metrics import classification_report
from visualization import plot_cost
from model_io import save_model, load_model
from feature_importance import feature_importance


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VALID_AGE_GROUPS = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
VALID_GENDERS = ["Male", "Female", "Joint", "Sex Not Available"]
VALID_LOAN_PURPOSE = ["p1", "p2", "p3", "p4"]
VALID_OCCUPANCY = ["pr", "sr", "ir"]


def train():

    file_path = os.path.join(BASE_DIR, "Loan_Default.csv")

    X_train, X_test, y_train, y_test = preprocess(file_path)

    print("\nTraining Model...\n")

    weights, bias, cost_history = train_model(
        X_train,
        y_train,
        learning_rate=1.92,
        iterations=2000
    )

    save_model(weights, bias)

    predictions = predict(X_test, weights, bias)

    classification_report(y_test, predictions)

    feature_names = np.load(
        os.path.join(BASE_DIR, "feature_names.npy"),
        allow_pickle=True
    ).tolist()
    feature_importance(weights, feature_names)

    plot_cost(cost_history)


def ask_choice(prompt, valid_options):
    """Keep asking until the user enters one of the valid training categories.
    This avoids silently producing all-zero dummy columns for a typo'd
    category, which would otherwise skew the prediction."""

    options_str = " / ".join(valid_options)

    while True:
        value = input(f"{prompt} [{options_str}] : ").strip()
        if value in valid_options:
            return value
        print(f"  Invalid value. Please choose one of: {options_str}")


def predict_customer():

    weights, bias = load_model()

    feature_names = np.load(
        os.path.join(BASE_DIR, "feature_names.npy"),
        allow_pickle=True
    ).tolist()

    with open(os.path.join(BASE_DIR, "fill_values.json")) as f:
        fill_values = json.load(f)

    with open(os.path.join(BASE_DIR, "norm_stats.json")) as f:
        norm_stats = json.load(f)

    print("\nEnter Customer Details")

    customer = {}

    customer["loan_amount"] = float(input("Loan Amount : "))
    customer["income"] = float(input("Income : "))
    customer["Credit_Score"] = float(input("Credit Score : "))
    customer["property_value"] = float(input("Property Value : "))
    customer["LTV"] = float(input("Loan To Value (LTV) : "))
    customer["dtir1"] = float(input("Debt To Income Ratio : "))
    customer["rate_of_interest"] = float(input("Interest Rate : "))
    customer["term"] = float(input("Loan Term : "))
    customer["age"] = ask_choice("Age Group", VALID_AGE_GROUPS)
    customer["Gender"] = ask_choice("Gender", VALID_GENDERS)
    customer["loan_purpose"] = ask_choice("Loan Purpose", VALID_LOAN_PURPOSE)
    customer["occupancy_type"] = ask_choice("Occupancy Type", VALID_OCCUPANCY)

    # Uses the SAME fill + normalization stats computed during training,
    # so the numbers the model sees at prediction time are on the same
    # scale as the numbers it was trained on.
    customer_array = preprocess_customer(
        customer,
        feature_names,
        fill_values,
        norm_stats
    )

    probability = predict_probability(
        customer_array,
        weights,
        bias
    )[0][0]

    print("\nProbability :", round(probability * 100, 2), "%")

    if probability >= 0.5:
        print("Prediction : Default")
    else:
        print("Prediction : Non Default")


if __name__ == "__main__":

    while True:

        print("\n")
        print("=" * 50)
        print("Loan Default Prediction System")
        print("=" * 50)

        print("1. Train Model")
        print("2. Predict Customer")
        print("3. Exit")

        choice = input("\nEnter Choice : ")

        if choice == "1":
            train()

        elif choice == "2":
            try:
                predict_customer()
            except FileNotFoundError:
                print("\nNo trained model found. Please train the model first (Option 1).")

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid Choice")
