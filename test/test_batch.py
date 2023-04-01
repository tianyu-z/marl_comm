# pass test with newest version of pettingzoo and tianshou
import numpy as np
from tianshou.data import Batch

b1 = Batch({"k1": {"sk1": [1, 2, 3], "sk2": np.zeros((3, 3))}, "k2": [1, 2, 3]})

print("b1")
print(b1)

sb1 = b1[[1]]
sb1.k2 = 4
print("sb1")
print(sb1)

b1.k2[[1]] = sb1.k2

print("b1")
print(b1)
