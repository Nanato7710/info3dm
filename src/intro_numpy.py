#%%
import numpy as np
# %%
a = np.ones((5, 1))
print(a)
# %%
a[1, 0] = 3.14
print(a)
# %%
b = a.copy().T
print(b)
# %%
dot_product = b @ a
print(dot_product)
# %%
x = np.random.rand(10, 1)
print(x)
# %%
A = np.random.normal(10, 2, (2, 5))
print(A)
# %%
a_column_2 = A[:, 1]
print(a_column_2)
# %%
a_column_3_4 = A[:, 2:4]
print(a_column_3_4)
# %%
B = np.random.rand(5, 2)
AB = A @ B
print(AB)
BA = B @ A
print(BA)