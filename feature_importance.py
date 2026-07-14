import numpy as np


def feature_importance(weights, feature_names):

    importance = np.abs(weights.flatten())

    indices = np.argsort(importance)[::-1]

    print("\nTop Important Features\n")

    for i in indices[:10]:

        print(f"{feature_names[i]:25} {importance[i]:.5f}")