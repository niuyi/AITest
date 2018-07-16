import numpy as np

mat = np.mat(([1, 2, 3], [4, 5, 6]))
print(mat)

# [[1 2 3]
#  [4 5 6]]

mat = np.zeros((2, 3))
print(mat)
#
# [[0. 0. 0.]
#  [0. 0. 0.]]

mat = np.ones((2, 3))
print(mat)
#
# [[1. 1. 1.]
#  [1. 1. 1.]]

mat = np.random.rand(2, 3)
print(mat)

# [[0.76812546 0.00899898 0.34970735]
#  [0.19858156 0.66888265 0.64210237]]

print(np.eye(3))
#
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]