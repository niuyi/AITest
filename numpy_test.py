import numpy as np

mat = np.mat(([1, 2, 3], [4, 5, 6]))
print(mat)
print(mat.shape) #(2, 3)

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

mat1 = np.mat(([1, 1, 1], [1, 1, 1]))
mat2 = np.mat(([2, 2, 2], [2, 2, 2]))

print(mat1 + mat2)
#
# [[3 3 3]
#  [3 3 3]]

mat3 = np.mat(([3, 3, 3], [3, 3, 3]))
mat2 = np.mat(([2, 2, 2], [2, 2, 2]))

print(mat3 - mat2)
#
# [[1 1 1]
#  [1 1 1]]

mat1 = np.mat(([1, 2], [3, 4], [5, 6]))
mat2 = np.mat(([3, 3], [2, 2], [1, 1]))

print(np.multiply(mat1, mat2))

# [[3 6]
#  [6 8]
#  [5 6]]

mat1 = np.mat(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
mat2 = np.mat(([1], [2], [3]))
print(mat1 * mat2)

# [[14]
#  [32]
#  [50]]

mat1 = np.mat(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
print(mat1.T)

# [[1 4 7]
#  [2 5 8]
#  [3 6 9]]

#行列式

mat1 = np.mat(([1, 2], [3, 4]), dtype=np.int)
print(np.linalg.det(mat1))

mat1 = np.mat(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
print(np.linalg.det(mat1))