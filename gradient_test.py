from numpy import *

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


def buildMat(dataSet):
	m,n=shape(dataSet)
	dataMat = zeros((m,n))
	dataMat[:,0] = 1
	dataMat[:,1:] = dataSet[:,:-1]
	return 	dataMat


def hardlim(dataSet):
	dataSet[nonzero(dataSet.A>0)[0]]=1
	dataSet[nonzero(dataSet.A<=0)[0]]=0
	return dataSet

Input = file2matrix("data/testSet.txt","\t")
target = Input[:,-1] #获取分类标签列表
[m,n] = shape(Input)

dataMat = buildMat(Input)
alpha = 0.001 # 步长
steps = 500  # 迭代次数

# print(dataMat)

weights = ones((n,1))# 初始化权重向量
# 主程序
for k in range(steps):
	gradient = dataMat*mat(weights) # 梯度
	output = hardlim(gradient)  # 硬限幅函数
	errors = target-output # 计算误差
	weights = weights + alpha*dataMat.T*errors

print(weights)	# 输出权重