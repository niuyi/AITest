from numpy import *
A = mat([[5, 5, 3, 0, 5, 5],[5, 0, 4, 0, 4, 4],[0, 3, 0, 5, 4, 5],[5, 4, 3, 3, 5, 5]])

# 手工分解求矩阵的svd
#公式 A = U*E*V.T
# U = A.A.T的特征向量组成的矩阵
#V.T = A.T*A的特征向量组成的矩阵

U = A*A.T
lamda,hU = linalg.eig(U) # hU:U特征向量
VT = A.T*A
eV,hVT = linalg.eig(VT)  # hVT:VT特征向量
hV = hVT.T
# print "hU:",hU
# print "hV:",hV
sigma = 	sqrt(lamda)         # 特征值
print("sigma:",sigma)
print('hU\n', hU)
print('hV\n', hV)


Sigma = zeros([shape(A)[0], shape(A)[1]])
U,S,VT = linalg.svd(A)
print('U\n', U)
print('S\n', S)
print('VT\n', VT)

# A = mat([
#     [-2, 1, 1],
#     [0, 2, 0],
#     [-4, 1, 3]
# ])
#
# a,b = linalg.eig(A)
#
# print(a)
# print(b)