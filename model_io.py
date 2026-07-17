import numpy as np

PROJECT_DIR = "C:/Users/msiindia/Desktop/loan_default_prediction/"
WEIGHTS_PATH = PROJECT_DIR + "weights.npy"
BIAS_PATH = PROJECT_DIR + "bias.npy"


# Save Model


def save_model(weights, bias):

    np.save(WEIGHTS_PATH, weights)
    np.save(BIAS_PATH, np.array([bias]))

    print("\nModel Saved Successfully!")



# Load Model

def load_model():

    weights = np.load(WEIGHTS_PATH)
    bias = np.load(BIAS_PATH)[0]

    print("\nModel Loaded Successfully!")

    return weights, bias
