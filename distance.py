import numpy as ny

# print('hello distance')
#
# mat = ny.mat([
#     [1,2,3],
#     [4,5,7]
# ])
#
# print(mat)
#
# mean0 = ny.mean(mat[0])
# mean1 = ny.mean(mat[1])
#
# print(mean0, mean1)
#
# dv0 = ny.std(mat[0])
# dv1 = ny.std(mat[1])
#
# print("dv0, dv1", dv0, dv1)
#
# #相关系数
# diff0 = mat[0] - mean0
# print(diff0)
#
# diff1 = mat[1] - mean1
# print(diff1)
#
# mul = ny.multiply(diff0, diff1)
# print("mul", mul)
#
# cor = (ny.mean(mul))/(dv0*dv1)
# print("cor", cor)
#
# #直接算法
# print(ny.corrcoef(mat))
#
# #协方差矩阵
#
# cov = ny.cov(mat)
# print(cov)
#
# cov_inv = ny.linalg.inv(cov)
# print(cov_inv)
#
# tp = mat.T[0] - mat.T[1]
# print(tp)
#
# print(mat.T)

#另一个例子

mat = ny.mat([
    [88, 96, 104], #身高
    [12, 14, 18] #体重
])

covinv = ny.linalg.inv(ny.cov(mat))
tp = mat.T[0] - mat.T[1]
distma = ny.sqrt(ny.dot(ny.dot(tp, covinv), tp.T))

print(distma)

mat = ny.mat([
    [88, 12],
    [96, 14],
    [104, 19]])

covinv = ny.linalg.inv(ny.cov(mat))
tp = mat[0] - mat[1]
distma = ny.sqrt(ny.dot(ny.dot(tp, covinv), tp.T))
print(distma)


