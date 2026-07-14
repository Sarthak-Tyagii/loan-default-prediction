import matplotlib.pyplot as plt


# Cost vs Iterations


def plot_cost(cost_history):

    plt.figure(figsize=(8,5))

    plt.plot(cost_history)

    plt.title("Cost vs Iterations")

    plt.xlabel("Iterations")

    plt.ylabel("Cost")

    plt.grid(True)

    plt.show()

