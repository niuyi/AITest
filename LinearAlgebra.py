from numpy import *

print('hello numpy')

myArray = [
    [11, 12, 13],
    [21, 22, 23]
]

myNDArray = array(myArray)

print(myNDArray)

# type <class 'numpy.ndarray'>
# ndim 2
# shape (2, 3)
# size 6
# dtype int32
# itemsize 4
# data <memory at 0x00000203A05FFB40>
print('type', type(myNDArray))
print('ndim', myNDArray.ndim)
print('shape', myNDArray.shape)
print('size', myNDArray.size)
print('dtype', myNDArray.dtype)
print('itemsize', myNDArray.itemsize)
print('data', myNDArray.data)
#
# zeros
#  [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]
# ones
#  [[1. 1. 1. 1.]
#  [1. 1. 1. 1.]
#  [1. 1. 1. 1.]]
# empty
#  [[1. 1. 1. 1.]
#  [1. 1. 1. 1.]
#  [1. 1. 1. 1.]]
print('zeros\n', zeros((3,4)))
print('ones\n', ones((3,4)))
print('empty\n', empty((3,4)))

# [10 15 20 25]
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
# [10. 15. 20. 25. 30.]
print(arange(10,30,5))
print(arange(15))
print(linspace(10,30,5))

a = array([
    [11, 12, 13],
    [21, 22, 23]
])

b = array([
    [1, 2, 3],
    [4, 5, 6]
])

c = array([5])

# a+b
#  [[12 14 16]
#  [25 27 29]]
# a-b
#  [[10 10 10]
#  [17 17 17]]
# a*2
#  [[22 24 26]
#  [42 44 46]]
# a*b
#  [[ 11  24  39]
#  [ 84 110 138]]
# a*c
#  [[ 55  60  65]
#  [105 110 115]]
# a<20
#  [[ True  True  True]
#  [False False False]]

print('a+b\n', a+b)
print('a-b\n', a-b)
print('a*2\n', a*2)
print('a*b\n', a*b) #elementwise!!
print('a*c\n', a*c) #elementwise!!
print('a<20\n', a < 20)

D = array([
    [1, 0, 3, -1],
    [2, 1, 0, 2]
])

E = array([
    [4, 1, 0],
    [-1, 1, 3],
    [2, 0, 1],
    [1, 3, 4]
])

# D dot E
#  [[ 9 -2 -1]
#  [ 9  9 11]]

print('D dot E\n', D.dot(E))
#
# sum D 8
# min D -1
# max D 3
# max D [2 1 3 2]
# max D [3 2]
# cumsum D axis=1
#  [[1 1 4 3]
#  [2 3 3 5]]

print('sum D', D.sum())
print('min D', D.min())
print('max D', D.max())
print('max D axis=0', D.max(axis=0))
print('max D axis=1', D.max(axis=1))
print('cumsum D axis=1\n', D.cumsum(axis=1))#cumulative 累加

E = array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])

# 11

# [ 1 11 21 31 41]

# [[10 11 12 13]
#  [20 21 22 23]]

# [[11]
#  [21]]

#[40 41 42 43]

print(E[1,1]) # 11,从0开始计算
print(E[0:5, 1]) #each row in the second column of E
print(E[1:3, :]) # each column in the second and third row of #
print(E[1:3, 1:2])
print(E[-1]) #When fewer indices are provided than the number of axes, the missing indices are considered complete slices

print('############################################################################################################')
F = array([[[  0,  1,  2],               # a 3D array (two stacked 2D arrays)
                [ 10, 12, 13]],
               [[100,101,102],
                [110,112,113]]])

print(F.shape) #(2, 2, 3)

print(F[1,...])
# [[100 101 102]
#  [110 112 113]]

print(F[...,2])
# [[  2  13]
#  [102 113]]

print(F[:,1,:])
# [[ 10  12  13]
#  [110 112 113]]

for element in F.flat: #展开所有元素
    print(element)

print('############################################################################################################')

print('E.T', E.T) #转置

G = array([[ 8.,  8.],
       [ 0.,  0.]])

H = array([[ 1.,  8.],
       [ 0.,  4.]])

print('vstack(G, H)\n', vstack((G, H)))
print('hstack(G, H)\n', hstack((G, H)))
#
# vstack(G, H)
#  [[8. 8.]
#  [0. 0.]
#  [1. 8.]
#  [0. 4.]]
# hstack(G, H)
#  [[8. 8. 1. 8.]
#  [0. 0. 0. 4.]]

G1 = array([4, 2])
H1 = array([2, 8])

# vstack((G1, H1))
#  [[4 2]
#  [2 8]]
# hstack((G1, H1))
#  [4 2 2 8]
# column_stack((G1, H1))
#  [[4 2]
#  [2 8]]
# G1[:,newaxis]
#  [[4]
#  [2]]
# H1[:,newaxis]
#  [[2]
#  [8]]
# vstack((G1[:,newaxis], H1[:,newaxis]))
#  [[4]
#  [2]
#  [2]
#  [8]]
# hstack((G1[:,newaxis], H1[:,newaxis]))
#  [[4 2]
#  [2 8]]
# column_stack((G1[:,newaxis], H1[:,newaxis])
#  [[4 2]
#  [2 8]]


print('vstack((G1, H1))\n', vstack((G1, H1)))
print('hstack((G1, H1))\n', hstack((G1, H1)))
print('column_stack((G1, H1))\n', column_stack((G1, H1)))

print('G1[:,newaxis]\n', G1[:,newaxis])
print('H1[:,newaxis]\n', H1[:,newaxis])

print('vstack((G1[:,newaxis], H1[:,newaxis]))\n', vstack((G1[:,newaxis], H1[:,newaxis])))
print('hstack((G1[:,newaxis], H1[:,newaxis]))\n', hstack((G1[:,newaxis], H1[:,newaxis])))
print('column_stack((G1[:,newaxis], H1[:,newaxis])\n', column_stack((G1[:,newaxis], H1[:,newaxis])))

# I [1 2 3]
# I1 [2 3]
# I [  1 100   3]
# I1 [100   3]
# I [   1  100 1000]
# I1 [ 100 1000]

I = array([1, 2, 3])
print('I', I)

I1 = I[1:3]
print('I1', I1)

I[1] = 100
print('I', I)
print('I1', I1)

I1[-1] = 1000
print('I', I)
print('I1', I1)

print('############################################################################################################')

J = array([
    [1, 2],
    [3, 4]
])

# [[1 2]
#  [3 4]]
# J.transpose()
#  [[1 3]
#  [2 4]]
# J.linalg.inv()
#  [[-2.   1. ]
#  [ 1.5 -0.5]]
# J.dot(linalg.inv(J))
#  [[1.0000000e+00 0.0000000e+00]
#  [8.8817842e-16 1.0000000e+00]]

print(J)
print('J.transpose()\n', J.transpose()) #转置
print('J.linalg.inv()\n', linalg.inv(J)) #求逆
print('J.dot(linalg.inv(J))\n', J.dot(linalg.inv(J)))

# eye(2)
#  [[1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]
#
# trace(eye(5))
#  5.0
#
# linalg.solve(J, y)
#  [[-3.]
#  [ 4.]]
# diag((1, 2, 3))
#  [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

print('eye(5)\n', eye(5))
print('trace(eye(5))\n', trace(eye(5))) #a11+a22+...+ann

y = array([
    [5],
    [7]
])

print('linalg.solve(J, y)\n', linalg.solve(J, y)) #求解 J*X = y

print('diag((1, 2, 3))\n', diag((1, 2, 3)))

K = array([
    [3, -1],
    [-1, 3]
])

print('K\n', K)
print('linalg.eig(K)\n', linalg.eig(K)) #特征值及特征向量

K = array([
    [-1, 1, 0],
    [-4, 3, 0],
    [1, 0, 2]
])

print('K\n', K)
print('linalg.eig(K)\n', linalg.eig(K))

L = array([
    [0, 1, -1, 1],
    [4, 1, 1,1],
    [3, 4, 2, 1],
    [16, 9, 3, 1]
])

print('linalg.det(L)', linalg.det(L)) #计算行列式


my_array = array([
    [[1,2],[3,4]],
    [[5,6],[7,8]],
    [[5, 6], [7, 8]]
])
print('array', my_array)
print('array', my_array.shape)


#ValueError: matrix must be 2-dimensional
# my_mat = mat([
#     [[1,2],[3,4]],
#     [[5,6],[7,8]],
#     [[5, 6], [7, 8]]
# ])
# print('mat', my_mat)

import numpy as np
print(np.zeros([2,3]))
print(np.ones([2,3]))
print(np.random.rand(3,4))
print(np.eye(3))
print(np.ones([3,3]) + np.eye(3))
print(5*np.ones([3,3]))
m1 = np.mat([2,3])
m2 = np.mat([4,5])
print(np.multiply(m1,m2))
# print(m1*m2)
print(np.mat([2,3]) * np.eye(2))

print(m1.T)
print(m2.transpose())
print(np.shape(m1))

m3 = np.mat([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(np.shape(m3))
print(m3[0])
print(m3.T[0])
print(m3[0:3])
print(np.linalg.det(m3[0:3]))
print(np.linalg.matrix_rank(m3))
print(np.linalg.norm(np.mat([1,2,3])))

