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
mat1 = ny.mat([
    [88.5, 96.8, 104.1],
    [12.54, 14.65, 16.64],
    [1, 2, 300]
]
)
#可以按照概率去理解，每行是某个制的可能取值范围
#
# print(ny.corrcoef(mat1))
# print(ny.corrcoef(mat1.T))

# print(ny.std([1,2,3]))
# print(ny.std([105,106,107]))
# #两组数据以相同的幅度抖动，认为其有相关性
# print(ny.corrcoef([[1,2,3],[105,106,107]])) #相关性是1

# mat = ny.mat([[1,2,3],[105,106,107]])
# print(ny.corrcoef(mat))
# print(ny.corrcoef(mat.T))
#矩阵和其转置相关性一样？？

# print(ny.corrcoef(mat1))

#
# #协方差矩阵
#

# print(ny.cov([[1,2,3], [3,5,6]]))
# print(ny.std([1,3]))#方差计算公式除以N-1，不是N
mat = ny.mat([
    [88.5, 96.8, 104.1, 111.3, 117.7, 124.0,130.0],
    [12.54, 14.65, 16.64, 18.98, 21.26,24.0,27.33]
]
)

#原方法
covinv = ny.linalg.inv(ny.cov(mat))
tp = mat.T[0] - mat.T[1]#计算这两个人数据之间的马氏距离
print("tp", tp)

distma = ny.sqrt(ny.dot(ny.dot(tp, covinv), tp.T))

print(distma) #[[1.22322662]]


# cov = ny.cov(mat)
# print(cov)
# print(ny.linalg.inv(cov))
#
# print(mat.T[0])
# print(mat.T[1])
#
# print(mat.T[0] - mat.T[1])
#
# cov_inv = ny.linalg.inv(cov)
# print(cov_inv)
#
# tp = mat.T[0] - mat.T[1]
# print(tp)
#
# print(mat.T)

#另一个例子

# mat = ny.mat([
#     [88, 96, 104], #身高
#     [12, 14, 18] #体重
# ])
#
# covinv = ny.linalg.inv(ny.cov(mat))
# tp = mat.T[0] - mat.T[1]
# distma = ny.sqrt(ny.dot(ny.dot(tp, covinv), tp.T))
#
# print(distma)
#
# mat = ny.mat([
#     [88, 12],
#     [96, 14],
#     [104, 19]])
#
# covinv = ny.linalg.inv(ny.cov(mat))
# tp = mat[0] - mat[1]
# distma = ny.sqrt(ny.dot(ny.dot(tp, covinv), tp.T))
# print(distma)


