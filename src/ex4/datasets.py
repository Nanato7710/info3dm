import numpy as np
import pandas as pd
from typing import Literal

Format = Literal["numpy", "pandas"]

def true_function(x: np.ndarray) -> np.ndarray:
    """
    >>> y = true_function(0)
    >>> print(float(y))
    0.0
    """
    return np.sin(np.pi*x*0.8)*10

def get_true_function_data(n_samples: int = 20, random_seed: int = 0, format: Format = "numpy") -> np.ndarray|pd.DataFrame:
    np.random.seed(random_seed)
    x = np.random.uniform(-1, 1, n_samples)
    y = true_function(x)
    if format == "pandas":
        return pd.DataFrame({"観測点": x, "真値": y})
    else:
        return x, y

def add_noise(y: np.ndarray|None, df: pd.DataFrame|None = None, noise_mean: float = 0, noise_std: float = 2.0, random_seed: int = 0, format: Format = "numpy") -> np.ndarray|pd.DataFrame:
    np.random.seed(random_seed)
    if format == "pandas":
        noise = 0.5*np.random.normal(noise_mean, noise_std, size=len(df))
        df["観測値"] = df["真値"] + noise
        return df
    else:
        noise = 0.5*np.random.normal(noise_mean, noise_std, size=len(y))
        return y + noise

def save_data(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index=False, sep="\t")

def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, sep="\t")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    x = np.linspace(-1, 1, 100)
    y = true_function(x)

    # data_x, data_y = get_true_function_data(n_samples=20, random_seed=0)
    # df = pd.DataFrame({"観測点": data_x, "真値": data_y})
    df = get_true_function_data(n_samples=20, random_seed=0, format="pandas")

    # noisy_y = add_noise(data_y, noise_mean=0, noise_std=2.0, random_seed=0)
    # df["観測値"] = noisy_y
    df = add_noise(None, df=df, noise_mean=0, noise_std=2.0, random_seed=0, format="pandas")

    df.to_csv("data.tsv", index=False, sep="\t")

    df1 = pd.read_csv("data.tsv", sep="\t")

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