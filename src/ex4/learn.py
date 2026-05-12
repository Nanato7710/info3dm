import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pickle

RANDOM_SEED = 0


def load_dataset():
    df = datasets.load_data("data.tsv")
    X = df["観測点"].to_numpy().reshape(-1, 1)
    y = df["観測値"].to_numpy().reshape(-1, 1)
    return X, y


def get_model():
    model = LinearRegression()
    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict_model(model, X):
    return model.predict(X)

def illustrate_model(model, X, y):
    true_function_x = np.linspace(-1, 1, 100)
    true_function_y = datasets.true_function(true_function_x)
    pred_function_y = model.predict(true_function_x.reshape(-1, 1))
    plt.plot(true_function_x, true_function_y, label="True Function", color="blue")
    plt.plot(true_function_x, pred_function_y, label="Predicted Function", color="red")
    plt.scatter(X, y, color="green", label="Data Points")
    plt.legend()
    plt.title("True Function vs Predicted Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.savefig("model_illustration.png")
    plt.show()


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

def save_model(model, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model(filepath):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model

def recovery_model(model_filepath, X_test, y_test):
    model = load_model(model_filepath)
    y_pred = predict_model(model, X_test)
    print("Predicted values from loaded model:", y_pred)
    illustrate_model(model, X_test, y_test)


if __name__ == "__main__":
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    model = get_model()
    model = train_model(model, X_train, y_train)

    mae = evaluate_model(model, X_test, y_test)
    print("Mean Absolute Error:", mae)

    illustrate_model(model, X, y)

    save_model(model, "model.pkl")

    recovery_model("model.pkl", X_test, y_test)