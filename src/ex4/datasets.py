import numpy as np

def true_function(x: np.ndarray) -> np.ndarray:
    """
    >>> y = true_function(0)
    >>> print(float(y))
    0.0
    """
    return np.sin(np.pi*x*0.8)*10

def get_true_function_data(n_samples: int = 20, random_seed: int = 0):
    np.random.seed(random_seed)
    x = np.random.uniform(-1, 1, n_samples)
    y = true_function(x)
    return x, y

def add_noise(y: np.ndarray, noise_mean: float = 0, noise_std: float = 2.0, random_seed: int = 0) -> np.ndarray:
    np.random.seed(random_seed)
    noise = np.random.normal(noise_mean, noise_std, size=y.shape)
    return y + noise

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    x = np.linspace(-1, 1, 100)
    y = true_function(x)

    data_x, data_y = get_true_function_data(n_samples=20, random_seed=0)
    df = pd.DataFrame({"観測点": data_x, "真値": data_y})

    noisy_y = add_noise(data_y, noise_mean=0, noise_std=2.0, random_seed=0)
    df["観測値"] = noisy_y

    plt.title("True Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()

    plt.plot(x, y, color="blue", label="True Function", )
    plt.legend()
    plt.savefig("ex1.1.png")

    plt.scatter(df["観測点"], df["真値"], color="red", label="True Values")
    plt.legend()
    plt.savefig("ex1.2.png")

    plt.scatter(df["観測点"], df["観測値"], color="green", label="Noisy Observations")
    plt.legend()
    plt.savefig("ex1.3.png")

    plt.show()