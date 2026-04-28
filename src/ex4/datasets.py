import numpy as np

def true_function(x: np.ndarray) -> np.ndarray:
    """
    >>> y = true_function(0)
    >>> print(float(y))
    0.0
    """
    return np.sin(np.pi*x*0.8)*10


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 100)
    y = true_function(x)

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True Function")
    plt.legend(["y = sin(pi * x * 0.8) * 10"])
    plt.grid()
    plt.savefig("ex1.1.png")
    plt.show()