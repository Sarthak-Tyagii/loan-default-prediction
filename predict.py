import os
import json
import numpy as np

from logistic_regression import predict_probability
from model_io import load_model
from preprocessing import preprocess_customer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



# Risk Level


def risk_level(probability):

    if probability < 0.30:
        return "LOW"

    elif probability <= 0.60:
        return "MEDIUM"

    else:
        return "HIGH"



# Loan Recommendation


def recommendation(probability):

    if probability >= 0.6:
        return "REJECT"

    return "APPROVE"


# ---------------------------------------
# Predict Customer (X must already be
# preprocessed: filled, encoded, normalized
# with the SAME stats used during training)
# ---------------------------------------

def predict_customer(X, weights, bias):

    probability = predict_probability(X, weights, bias)

    probability = float(probability[0][0])

    print("\n" + "=" * 45)
    print("Loan Risk Assessment")
    print("=" * 45)

    print(f"Probability of Default : {probability*100:.2f}%")

    print(f"Risk Level             : {risk_level(probability)}")

    print(f"Recommendation         : {recommendation(probability)}")

    print("=" * 45)

    return probability


# Standalone CLI entry point


def _load_artifacts():

    weights, bias = load_model()

    feature_names = np.load(
        os.path.join(BASE_DIR, "feature_names.npy"),
        allow_pickle=True
    ).tolist()

    with open(os.path.join(BASE_DIR, "fill_values.json")) as f:
        fill_values = json.load(f)

    with open(os.path.join(BASE_DIR, "norm_stats.json")) as f:
        norm_stats = json.load(f)

    return weights, bias, feature_names, fill_values, norm_stats


if __name__ == "__main__":

    weights, bias, feature_names, fill_values, norm_stats = _load_artifacts()

    print("\nEnter Customer Details")

    customer = {
        "loan_amount": float(input("Loan Amount : ")),
        "income": float(input("Income : ")),
        "Credit_Score": float(input("Credit Score : ")),
        "property_value": float(input("Property Value : ")),
        "LTV": float(input("Loan To Value (LTV) : ")),
        "dtir1": float(input("Debt To Income Ratio : ")),
        "rate_of_interest": float(input("Interest Rate : ")),
        "term": float(input("Loan Term : ")),
        "age": input("Age Group (e.g. 25-34) : ").strip(),
        "Gender": input("Gender (Male/Female/Joint/Sex Not Available) : ").strip(),
        "loan_purpose": input("Loan Purpose (p1/p2/p3/p4) : ").strip(),
        "occupancy_type": input("Occupancy Type (pr/sr/ir) : ").strip(),
    }

    X = preprocess_customer(customer, feature_names, fill_values, norm_stats)

    predict_customer(X, weights, bias)
