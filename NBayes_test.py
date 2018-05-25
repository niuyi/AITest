import sys
import os
import time
from numpy import *
import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him', 'my'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


class NBayes(object):
    def __init__(self):
        self.vocabulary = []  # 词典,去重后的所有词，这里是1*32
        self.idf = 0  # 词典的idf权值向量，1×32，每个词在几个句子中出现，逆文档频率 Inverse Document Frequency
        self.tf = 0  # 训练集的权值矩阵，6×32，每个句子中每个词出现的次数，词频 Term Frequency，缩写为TF
        self.tdm = 0  # P(x|yi)，2×32，每个词在每个分类中出现的次数/这个分类所有词的数量
        self.Pcates = {}  # P(yi)--是个类别字典,每个类别的概率，这里0类0.5，1类0.5
        self.labels = []  # 对应每个文本的分类，是个外部导入的列表
        self.doclength = 0  # 训练集文本数
        self.vocablen = 0  # 词典词长
        self.testset = 0  # 测试集

    def train_set(self, trainset, classVec):
        self.cate_prob(classVec)  # 计算每个分类在数据集中的概率：P(yi)  {0: 0.5, 1: 0.5}
        self.doclength = len(trainset)  # 6
        print('doclength', self.doclength)
        tempset = set()
        [tempset.add(word) for doc in trainset for word in doc]  # 生成词典

        self.vocabulary = list(tempset)
        self.vocablen = len(self.vocabulary)
        print('vocablen', self.vocablen)  # 32

        self.calc_wordfreq(trainset)
        # self.calc_tfidf(trainset)  # 生成tf-idf权值，优化步骤
        self.build_tdm()  # 按分类累计向量空间的每维值：P(x|yi)

    # 计算每个分类在数据集中的概率：P(yi)
    def cate_prob(self, classVec):
        self.labels = classVec
        labeltemps = set(self.labels)  # 获取全部分类
        for labeltemp in labeltemps:
            # 统计列表中重复的值：self.labels.count(labeltemp)
            self.Pcates[labeltemp] = float(self.labels.count(labeltemp)) / float(len(self.labels))

        print(self.Pcates)

    # 生成普通的词频向量
    def calc_wordfreq(self, trainset):
        self.idf = np.zeros([1, self.vocablen])  # 1*词典数 1*32
        self.tf = np.zeros([self.doclength, self.vocablen])  # 训练集文件数*词典数  6*32
        for indx in range(self.doclength):  # 遍历所有的文本
            for word in trainset[indx]:  # 遍历文本中的每个词
                self.tf[indx, self.vocabulary.index(word)] += 1  # 找到文本的词在字典中的位置+1
                #j统计每个句子中，每个词出现的次数
            for signleword in set(trainset[indx]):
                #统计每个词在几个句子中出现，因为上一句加了set，已经对每个句子去重复
                self.idf[0, self.vocabulary.index(signleword)] += 1

    # 按分类累计向量空间的每维值：P(x|yi)
    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates), self.vocablen])  # 类别行*词典列
        sumlist = np.zeros([len(self.Pcates), 1])  # 统计每个分类的总值
        print('tf', self.tf)
        for indx in range(self.doclength):
            self.tdm[self.labels[indx]] += self.tf[indx]  # 将同一类别的词向量空间值加总
            #统计每个词在0类或1类中出现的次数
           # sumlist[self.labels[indx]] = np.sum(self.tdm[self.labels[indx]])  # 统计每个分类的总值--是个标量，统计0类或1类中所有词的数量
        sumlist[0] = sum(self.tdm[0])
        sumlist[1] = sum(self.tdm[1])

        print('sumList', sumlist)
        self.tdm = self.tdm / sumlist  # P(x|yi)

        # 生成 tf-idf
    def calc_tfidf(self, trainset):
        self.idf = np.zeros([1, self.vocablen])# 1*词典数 1*32
        self.tf = np.zeros([self.doclength, self.vocablen])# 训练集文件数*词典数  6*32
        for indx in range(self.doclength):
            for word in trainset[indx]:
                self.tf[indx, self.vocabulary.index(word)] += 1
            # 消除不同句长导致的偏差
            self.tf[indx] = self.tf[indx] / float(len(trainset[indx]))
            for signleword in set(trainset[indx]):
                self.idf[0, self.vocabulary.index(signleword)] += 1
        self.idf = np.log(float(self.doclength) / self.idf)
        self.tf = np.multiply(self.tf, self.idf)  # 矩阵与向量的点乘

    # 测试集映射到当前词典
    def map2vocab(self, testdata):
        self.testset = np.zeros([1, self.vocablen])
        for word in testdata:
            self.testset[0, self.vocabulary.index(word)] += 1

    # 输出分类类别
    def predict(self, testset):
        if np.shape(testset)[1] != self.vocablen:
            print
            "输入错误"
            exit(0)

        predvalue = 0
        predclass = ""

        for tdm_vect, keyclass in zip(self.tdm, self.Pcates):
            # P(x|yi)P(yi)
            # print(testset * tdm_vect * self.Pcates[keyclass])
            temp = np.sum(testset * tdm_vect * self.Pcates[keyclass])
            # print(temp)
            if temp > predvalue:
                predvalue = temp
                predclass = keyclass
        return predclass




dataSet, listClasses = loadDataSet()
nb = NBayes()
nb.train_set(dataSet, listClasses)
nb.map2vocab(['ate', 'my','stupid'])
print(nb.predict(nb.testset))
