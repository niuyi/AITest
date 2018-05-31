print('bikMean')

# -*- coding: utf-8 -*-
# Filename : 02kMeans1.py

from numpy import *
import numpy as np

def file2matrix(path, delimiter):
    recordlist = []
    fp = open(path, "rb")  # 读取文件内容
    content = fp.read()
    fp.close()
    rowlist = content.splitlines()  # 按行转换为一维表
    # 逐行遍历
    # 结果按分隔符分割为行向量
    for row in rowlist:
        if(row.strip()):
            line = [float(s) for s in row.decode().split(delimiter)]
            recordlist.append(line)

    # recordlist = [ row.decode().split(delimiter) for row in rowlist if row.strip()]

    return mat(recordlist)  # 返回转换后的矩阵形式

# 从文件构建的数据集
dataMat = file2matrix("data/4k2_far.txt","\t")
dataSet = mat(dataMat[:,1:]) # 转换为矩阵形式


k = 4 # 分类数
m = shape(dataSet)[0]
# 初始化第一个聚类中心: 每一列的均值
centroid0 = mean(dataSet, axis=0).tolist()[0]
centList =[centroid0] # 把均值聚类中心加入中心表中
print('len(centList)', len(centList))

print('centList', centList)
# 初始化聚类距离表,距离方差:
ClustDist = mat(zeros((m,2)))

eps = 1.0e-6
def distEclud(vecA, vecB):
	return linalg.norm(vecA-vecB)+eps  #防止除以0

# 随机生成聚类中心
def randCenters(dataSet, k):
    n = shape(dataSet)[1]
    clustercents = mat(zeros((k,n)))# 初始化聚类中心矩阵:k*n
    for col in range(n):
        mincol = min(dataSet[:,col]); maxcol = max(dataSet[:,col])
        # random.rand(k,1): 产生一个0~1之间的随机数向量：k,1表示产生k行1列的随机数
        clustercents[:,col] = mat(mincol + float(maxcol - mincol) * random.rand(k,1))
    return clustercents

# KMeans 主函数
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCenters):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            distlist =[ distMeas(centroids[j,:],dataSet[i,:]) for j in range(k) ]
            minDist = min(distlist)
            minIndex = distlist.index(minDist)
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

for j in range(m):
	ClustDist[j,1] = distEclud(centroid0,dataSet[j,:])**2
# print('ClustDist', ClustDist)
#依次生成k个聚类中心
while (len(centList) < k):
    lowestSSE = inf # 初始化最小误差平方和。核心参数，这个值越小就说明聚类的效果越好。inf无限大
    # 遍历cenList的每个向量
    #----1. 使用ClustDist计算lowestSSE，以此确定:bestCentToSplit、bestNewCents、bestClustAss----#
    for i in range(len(centList)):
        print('i', i)
        ptsInCurrCluster = dataSet[nonzero(ClustDist[:,0].A==i)[0],:]
        print('ptsInCurrCluster', ptsInCurrCluster)
        # 应用标准kMeans算法(k=2),将ptsInCurrCluster划分出两个聚类中心,以及对应的聚类距离表
        centroidMat,splitClustAss = kMeans(ptsInCurrCluster, 2)
        # centroidMat　中心点　２＊２
        # splitClustAss m*2, index, 距离
        # 计算splitClustAss的距离平方和
        sseSplit = sum(splitClustAss[:,1]) #分割后所有距离的和
        # 计算ClustDist[ClustDist第1列!=i的距离平方和
        #ClustDist[:,0].A
        print('ClustDist[:,0].A\n', ClustDist[:,0].A)
        sseNotSplit = sum(ClustDist[nonzero(ClustDist[:,0].A!=i)[0],1]) #所有不以ｉ为中心的距离和

        print('sseNotSplit', sseNotSplit)
        if (sseSplit + sseNotSplit) < lowestSSE: # 算法公式: lowestSSE = sseSplit + sseNotSplit
            bestCentToSplit = i                 # 确定聚类中心的最优分隔点
            bestNewCents = centroidMat          # 用新的聚类中心更新最优聚类中心
            bestClustAss = splitClustAss.copy() # 深拷贝聚类距离表为最优聚类距离表
            lowestSSE = sseSplit + sseNotSplit  # 更新lowestSSE
        print('bestNewCents\n', bestNewCents)
    # 回到外循环
    #----2. 计算新的ClustDist----#
    # 计算bestClustAss 分了两部分:
    # 第一部分为bestClustAss[bIndx0,0]赋值为聚类中心的索引
    # print('bestClustAss before\n', bestClustAss)
    #找到最优的组，这个组内部被分为２部分，以０为核心，以１为核心，
    #把以１为核心的指向一个新的值，len(centList)就是增加了一个新值
    #以０为核心的，只想这个组的index,
    bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
    # 第二部分 用最优分隔点的指定聚类中心索引
    bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit

    # 以上为计算bestClustAss
    # 更新ClustDist对应最优分隔点下标的距离，使距离值等于最优聚类距离对应的值
    #以上为计算ClustDist

    #----3. 用最优分隔点来重构聚类中心----#
    # 覆盖: bestNewCents[0,:].tolist()[0]附加到原有聚类中心的bestCentToSplit位置
    # 增加: 聚类中心增加一个新的bestNewCents[1,:].tolist()[0]向量
    centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] #更新为群内部的新的中心点中的第一个，以0为中心
    centList.append(bestNewCents[1,:].tolist()[0])#内部以1为中心的添加在后面
    print('################################################################')
    print('bestClustAss\n' , bestClustAss)
    print('ClustDist\n' , ClustDist)
    ClustDist[nonzero(ClustDist[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    #所有原来指向被分组的点，整体替换成新点

    print('ClustDist\n', ClustDist)
    # 以上为计算centList
# color_cluster(ClustDist[:,0:1],dataSet,plt)
# print "cenList:",mat(centList)
# # print "ClustDist:", ClustDist
# # 绘制聚类中心图形
# drawScatter(plt,mat(centList),size=60,color='red',mrkr='D')
#
# plt.show()

ptsInClust = mat([
    [1,2],
    [3,6],
    [5,4]
]
)

test = [0,1]

testmat = mat([
    [0, 100],
    [1, 10000]
])

ptsInClust[test] = testmat

print('ptsInClust', ptsInClust)

# print('sum', sum(ptsInClust[test, 1]))
#
# centroid0 = mean(ptsInClust, axis=0).tolist()[0]
#
# print('centroid0', centroid0) #centroid0 [3.0, 4.0]