from math import log
import numpy as np

#从文档中读取数据，每条数据转成列表的形式
def readData(path):
    dataList = []
    with open(path,'r') as f:
        dataSet = f.readlines() #一行一行读
    
    for d in dataSet:
        d = d[:-1]
        d = d.split(',')
        dataList.append(d)
    
    return dataList

#将数据集划分为训练集和测试集
def splitTestData(dataList,testnum):
    trainData = []
    testData = []
    dataNum = len(dataList)
    print("dataNum: ",dataNum)
    pred_ind = np.random.randint(0,dataNum,testnum) #在总数据里取一部分做数据集，一部分做测试集

    for d in pred_ind:
        testData.append(dataList[d])
    for d in range(dataNum):
        if d not in pred_ind:
            trainData.append(dataList[d])

    print("dataSetNum:",dataNum,len(trainData),len(testData))
    return trainData,testData

# Class Values:
# unacc, acc, good, vgood

# Attributes:
# buying: vhigh, high, med, low.
# maint: vhigh, high, med, low.
# doors: 2, 3, 4, 5more.
# persons: 2, 4, more.
# lug_boot: small, med, big.
# safety: low, med, high.
Cls = {'unacc':0, 'acc':1, 'good':2, 'vgood':3}   #分类值映射
#特征值映射，共6个特征值，每个特征表示为X[i]，X[i][xiv]表示特征Xi的取值。
X = [
        {'vhigh':0, 'high':1, 'med':2, 'low':3},
        {'vhigh':0, 'high':1, 'med':2, 'low':3},
        {'2':0, '3':1, '4':2, '5more':3},
        {'2':0, '4':1, 'more':2},
        {'small':0, 'med':1, 'big':2},
        {'low':0, 'med':1, 'high':2}
    ]

#计算个类的个数
def CountEachClass(dataSet):
    numEachClass = [0]*len(Cls)  #列表初始化
    for d in dataSet:
        numEachClass[Cls[d[-1]]] += 1
    #print("numEachClass",numEachClass)  # 各类的个数
    return numEachClass

# 利用PPT-P44的熵计算公式进行熵的计算（熵：衡量数据集的纯度）
def caculateEntropy(dataSet):
    NumEachClass = CountEachClass(dataSet)
    dataNum = len(dataSet)
    #print("dataNum: ",dataNum)
    ent = 0
    # print("NumEachClass:\n",NumEachClass)
    for numC in NumEachClass:
        #print("numC:",numC)
        temp = numC/dataNum
        if(temp != 0):
            ent -= temp * log(temp,2) # 熵计算公式 PPT-P44
    return ent

def splitData(dataSet,xi):
    subDataSets = [ [] for i in range(len(X[xi]))]  #子数据集列表初始化
    #print("子数据集：",subDataSets)
    for d in dataSet:
        subDataSets[ X[xi][d[xi]] ].append(d)
    
    return subDataSets

# 利用PPT-P47的熵计算公式进行属性A的熵值计算
# 计算信息增益，选择信息收益最大的属性进行分裂
# return 信息收益
def calGain(dataset,xi): #xi：属性xi
    res = 0
    ent = caculateEntropy(dataset) # 熵
    subDataSet = splitData(dataset,xi)
    for xivDataSet in subDataSet:
        if(xivDataSet):
            res += len(xivDataSet)/len(dataset) * caculateEntropy(xivDataSet) # 带入计算公式
    
    # 属性的信息收益为：原有熵值 - 分裂后的熵值
    return ent - res

#获得最大的信息增益值和对应的特征序号
def getMaxGain(dataSet,usedX=[]):   
    gains = []
    for xi in range(len(X)):
        if(xi not in usedX):
            gains.append(calGain(dataSet,xi))
        else:
            gains.append(0)
    
    print("gains: ",gains)
    mg = max(gains) # 获取最大收益
    mx = gains.index(mg) # 获取最大收益索引
    print("gains.index(max(gains)): ",gains.index(mg))
    return mx,mg

# 以字典的结构构建决策树
def createTree(dataSet,r,usedX=[]):   
    if (len(dataSet)==0):
        return {}     # 空树
    tree = {}
    numEachClass = CountEachClass(dataSet)
    print("numEachClass: ",numEachClass)
    c = numEachClass.index(max(numEachClass))
    tree['class'] = c  # 该树各分类中数据最多的类，记为该根节点的分类
    mx,mg = getMaxGain(dataSet,usedX)
    print("max gain:",mg)
    if len(usedX) == len(X) or numEachClass[c] == len(dataSet) or mg < r:
        tree['factureX'] = -1    #不在继续分支，即为叶节点
        return tree

    else:
        tree['factureX']= mx  #记录该根节点用于划分的特征
        subDataSet = splitData(dataSet, mx)  #用该特征的值划分子集，用于构建子树
        for xiv in range(len(X[mx])):
            xivDataSet = subDataSet[xiv]
            newusedX = usedX.copy()
            newusedX.append(mx)
            tree[xiv] = createTree(xivDataSet,r,newusedX) # 递归构建一棵树

    return tree

def classify(tree,data):
    xi = tree['factureX']  #根节点用于划分子树的特征
    if(xi>=0):
        subtree = tree[X[xi][data[xi]]]
        if subtree=={}: #节点没有该子树
            return tree['class']  #以该节点的分类作为数据的分类
        return classify(subtree,data)  #否则遍历子树
    else: #叶节点
        return tree['class']

# 测试：
testNum = 100
err = 0
right = 0

dataSet = readData('car.data.txt')
trainDataSet,testDataSet = splitTestData(dataSet,testNum)
tree = createTree(trainDataSet,0.2)

for d in testDataSet:
    c = classify(tree,d)
    if c ==Cls[d[-1]]:
        right +=1
    else:
        err +=1
    print('fact:',Cls[d[-1]]," predict:",c)

print("total:",testNum,"error:",err,"right:",right)
print("accurate：",1 - err/testNum)