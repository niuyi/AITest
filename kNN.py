print('kNN test')
import numpy as np




def createDataSet():
    group = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(testdata, dataSet, labels, k):
    # 返回样本集的行数
    dataSetSize = dataSet.shape[0]
    print('dataSetSize', dataSetSize)
    # 计算测试集与训练集之间的距离：标准欧氏距离
    # 1.计算测试项与训练集各项的差
    print('testData', testdata)
    print('tile', np.tile(testdata, (dataSetSize, 1)))

    # tile
    # [
    #   [0.2 0.2]
    #   [0.2 0.2]
    #   [0.2 0.2]
    #   [0.2 0.2]
    # ]

    diffMat = np.tile(testdata, (dataSetSize, 1)) - dataSet
    print('diffMat', diffMat)
    # 2.计算差的平方和
    sqDiffMat = diffMat ** 2

    print('sqDiffMat', sqDiffMat)
    # 3.按列求和
    sqDistances = sqDiffMat.sum(axis=1) #每行相加
    print('sqDistances', sqDistances)
    # 4.生成标准化欧氏距离
    distances = sqDistances ** 0.5
    print('distances', distances)
    # print distances
    # 5.根据生成的欧氏距离大小排序,结果为索引号
    sortedDistIndicies = distances.argsort() #从小到大的序号 sortedDistIndicies [3 2 1 0]

    print('sortedDistIndicies', sortedDistIndicies)
    classCount = {}
    # 获取欧氏距离的前三项作为参考项
    for i in range(k):  # i = 0~(k-1)
        # 按序号顺序返回样本集对应的类别标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # 为字典classCount赋值,相同key，其value加1
        # key:voteIlabel，value: 符合voteIlabel标签的训练集数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    print('classCount', classCount)
    # 对分类字典classCount按value重新排序
    # sorted(data.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 该句是按字典值排序的固定用法
    # classCount.iteritems()：字典迭代器函数
    # key：排序参数；operator.itemgetter(1)：多级排序
    sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
    # 返回序最高的一项
    return sortedClassCount[0][0]


k = 3
testdata = [0.2, 0.2]
dataSet, labels = createDataSet()

print('result', classify(testdata, dataSet, labels, k))
