import datasets
from sklearn.linear_model import LinearRegression


df = datasets.load_data("data.tsv")

model = LinearRegression()