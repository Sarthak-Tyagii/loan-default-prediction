import numpy as np
import pandas as pd
import json


# Load Dataset
def load_data(file_path):

    data = pd.read_csv(file_path)

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)
    print("Shape :", data.shape)

    return data


def select_features(data):

    selected_features = [

        "loan_amount",
        "income",
        "Credit_Score",
        "property_value",
        "LTV",
        "dtir1",
        "rate_of_interest",
        "term",
        "age",
        "Gender",
        "loan_purpose",
        "occupancy_type",
        "Status"

    ]

    data = data[selected_features]

    return data


# Drop Unnecessary Columns
def drop_columns(data):

    if "ID" in data.columns:
        data = data.drop(columns=["ID"])

    return data


# Fill Missing Values
# NOTE: the fill values (median/mode) are computed on the TRAINING data
# and returned so the exact same values can be reused at prediction time.
def handle_missing_values(data, fill_values=None):

    numerical_columns = data.select_dtypes(include=np.number).columns
    categorical_columns = data.select_dtypes(exclude=np.number).columns

    is_training = fill_values is None
    if is_training:
        fill_values = {}

    for col in numerical_columns:
        if is_training:
            fill_values[col] = float(data[col].median())
        data[col] = data[col].fillna(fill_values[col])

    for col in categorical_columns:
        if is_training:
            fill_values[col] = data[col].mode()[0]
        data[col] = data[col].fillna(fill_values[col])

    return data, fill_values


# Encode Categorical Features
def encode_categorical(data):

    data = pd.get_dummies(data, drop_first=True)

    # Convert True/False columns to 0/1
    bool_cols = data.select_dtypes(include="bool").columns
    data[bool_cols] = data[bool_cols].astype(int)

    return data


# Normalize Features
# NOTE: min/max are computed on the TRAINING data only and returned so the
# exact same min/max can be applied at prediction time. Previously this
# function computed a fresh min/max every call and never saved it, so
# predictions used unnormalized raw values -> completely wrong probabilities.
def normalize(data, stats=None):

    is_training = stats is None
    if is_training:
        stats = {}

    for col in data.columns:

        if col == "Status":
            continue

        if is_training:
            minimum = data[col].min()
            maximum = data[col].max()
            stats[col] = (float(minimum), float(maximum))
        else:
            minimum, maximum = stats.get(col, (0.0, 1.0))

        if maximum != minimum:
            data[col] = (data[col] - minimum) / (maximum - minimum)
        else:
            data[col] = 0.0

    return data, stats


# Train Test Split
def train_test_split(data, test_size=0.2):

    data = data.sample(frac=1, random_state=42)

    split = int(len(data) * (1 - test_size))

    train = data[:split]
    test = data[split:]

    X_train = train.drop("Status", axis=1).to_numpy()
    y_train = train["Status"].to_numpy().reshape(-1, 1)

    X_test = test.drop("Status", axis=1).to_numpy()
    y_test = test["Status"].to_numpy().reshape(-1, 1)

    return X_train, X_test, y_train, y_test


# Complete Pipeline (TRAINING)
# Saves feature_names.npy, normalization stats, and missing-value fill
# stats to disk so predict-time preprocessing can be made identical.
def preprocess(file_path):

    data = load_data(file_path)

    data = select_features(data)

    data = drop_columns(data)

    data, fill_values = handle_missing_values(data)

    data = encode_categorical(data)

    data, norm_stats = normalize(data)

    feature_names = data.drop("Status", axis=1).columns.tolist()

    np.save("feature_names.npy", feature_names)

    with open("fill_values.json", "w") as f:
        json.dump(fill_values, f, indent=2)

    with open("norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    return train_test_split(data)


# Preprocess a SINGLE customer record (PREDICTION)
# Applies the exact same missing-value fill and normalization stats
# that were computed during training, then aligns columns to the
# training feature order.
def preprocess_customer(customer_dict, feature_names, fill_values, norm_stats):

    data = pd.DataFrame([customer_dict])

    # Fill any missing/blank fields using the training fill values
    for col, val in fill_values.items():
        if col == "Status":
            continue
        if col not in data.columns:
            data[col] = val
        else:
            data[col] = data[col].fillna(val) if data[col].isna().any() else data[col]

    data = encode_categorical(data)

    # Align to training columns (adds any missing dummy columns as 0,
    # drops anything extra, and puts columns in the right order)
    data = data.reindex(columns=feature_names, fill_value=0)

    # Apply the SAME min/max normalization used during training
    for col in data.columns:
        minimum, maximum = norm_stats.get(col, (0.0, 1.0))
        if maximum != minimum:
            data[col] = (data[col].astype(float) - minimum) / (maximum - minimum)
        else:
            data[col] = 0.0

    data = data.astype(float)

    return data.to_numpy(dtype=np.float64)
