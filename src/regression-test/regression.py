import numpy as np


class LinearRegression:
    x = None
    theta = None
    y = None

    def fit(self, x, y):
        temp = np.linalg.inv(np.dot(x.T, x))
        self.theta = np.dot(np.dot(temp, x.T), y)

    def predict(self, x):
        pass

    def score(self, x, y):
        pass