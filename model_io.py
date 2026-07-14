import os
import numpy as np

# Project root (one level up from utils/), so saving/loading works no
# matter which directory the script is launched from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights.npy")
BIAS_PATH = os.path.join(BASE_DIR, "bias.npy")


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