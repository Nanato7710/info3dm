import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pickle


df = datasets.load_data("data.tsv")
X = df["観測点"].to_numpy().reshape(-1, 1)
y = df["観測値"].to_numpy().reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predicted values:", y_pred)

true_function_x = np.linspace(-1, 1, 100)
true_function_y = datasets.true_function(true_function_x)
pred_function_y = model.predict(true_function_x.reshape(-1, 1))
plt.plot(true_function_x, true_function_y, label="True Function", color="blue")
plt.plot(true_function_x, pred_function_y, label="Predicted Function", color="red")
plt.scatter(df["観測点"], df["観測値"], color="green", label="Data Points")
plt.legend()
plt.title("True Function vs Predicted Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.savefig("ex1.10.png")
plt.show()

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

loaded_y_pred = loaded_model.predict(X_test)
print("Predicted values from loaded model:", loaded_y_pred)

loaded_function_y = loaded_model.predict(true_function_x.reshape(-1, 1))
plt.plot(true_function_x, true_function_y, label="True Function", color="blue")
plt.plot(true_function_x, loaded_function_y, label="Loaded Model Predicted Function", color="orange")
plt.scatter(df["観測点"], df["観測値"], color="green", label="Data Points")
plt.legend()
plt.title("True Function vs Loaded Model Predicted Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.savefig("ex1.14.png")
plt.show()