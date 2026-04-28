import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score


df = datasets.load_data("data.tsv")
X_train, X_test, y_train, y_test = train_test_split(df[["観測点"]], df["観測値"], test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)