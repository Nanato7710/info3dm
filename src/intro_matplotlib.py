# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
datasets = np.array([[4,7], [8,10], [13,11], [17,14]])
x = datasets[:, 0]
y = datasets[:, 1]
# %%
plt.scatter(x, y, color="black", label="dataset")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
plt.show()
# %%
def function(x):
    return (x-10)**2

x2 = np.linspace(4, 17, 5)
y2 = function(x2)
plt.plot(x2, y2, color="blue", linewidth=3, label="plot example")
plt.show()
# %%
x2 = np.linspace(4, 17, 100)
y2 = function(x2)
plt.plot(x2, y2, color="blue", linewidth=3, label="plot example")
plt.show()
# %%
plt.scatter(x, y, color="black", label="dataset")
plt.plot(x2, y2, color="blue", linewidth=3, label="plot example")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
plt.show()
# %%
import japanize_matplotlib
plt.scatter(x, y, color="black", label="データセット")
plt.plot(x2, y2, color="blue", linewidth=3, label="プロット例")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
plt.show()