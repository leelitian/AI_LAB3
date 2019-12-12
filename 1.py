import numpy as np
from enum import Enum

#从文档中读取数据，每条数据转成列表的形式
def readData(path):
    dataList = []
    with open(path,'r') as f:
        dataSet = f.readlines()

    for d in dataSet:
        d = d[:-1]
        d = d.split(',')
        dataList.append(d)

    return dataList

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
# 特征值映射，共6个特征值，每个特征表示为X[i]，X[i][xiv]表示特征Xi的取值。
X = [
        {'vhigh':0, 'high':1, 'med':2, 'low':3},
        {'vhigh':0, 'high':1, 'med':2, 'low':3},
        {'2':0, '3':1, '4':2, '5more':3},
        {'2':0, '4':1, 'more':2},
        {'small':0, 'med':1, 'big':2},
        {'low':0, 'med':1, 'high':2}
    ]

#训练模型，生成概率矩阵即生成p{Y=yi} 和 p{Xi=xiv|Y=yi}
def NBtrain(labelData):
    datanum = len(labelData)
    print("数据量: ",datanum)
    Arr = np.zeros((4,6,4))   # Arr[y][xi][xiv]表示在分类y的条件下，特征Xi取值为xiv的数量
    
    for d in labelData:
        y = Cls[d[-1]]     # 取分类的映射值
        for i in range(len(d)-1):
            v = X[i][d[i]]    # 取每个特征的映射值
            Arr[y][i][v] += 1 # 计数

    probXCY = np.zeros((4,6,4))  # probXCY[y][xi][xiv]表示在分类y的条件下，特征Xi取值为xiv的概率即 p{Xi=xiv|Y=y}
    numY = []         # 分类为yi的数量
    probY =[]         # 分类为yi的概率

    for y in Cls.values():
        numY.append(np.sum(Arr[y][0]))
        print("check",Arr[y][0])
        probY.append( (numY[y]+1)/(datanum+len(Cls)) ) 
        print("numY[y]+1：",numY[y]+1) # ***********************************
        print("datanum+len(Cls)：",datanum+len(Cls)) # ***********************************
        print("y=",y,","," numY[y]=",numY[y], ",","probY[y]=",probY[y])
        for xi in range(len(X)):
            s = len(X[xi])    #特征Xi的值的个数
            for xiv in X[xi].values():
                probXCY[y][xi][xiv] = (Arr[y][xi][xiv]+1)/(numY[y]+s)   #做拉普拉斯平滑避免概率值为0的情况
                print("probXCY[y][xi][xiv]=",probXCY[y][xi][xiv])

    print("Arr:\n",Arr)
    print("probXCY:\n",probXCY)
    return probXCY,probY

def NBclassify(probXCY,probY,predData):
    unknowData = predData
    datanum = len(unknowData)
    print("len(unknowData)",datanum)
    YofX = []    #记录数据的分类
    diffNum = 0  #记录分类结果与实际不同的数量
    for d in unknowData:
        probyCx = []    #记录p{Y=yi|X[...]=x[...]}
        for y in Cls.values():
            p = 10**5   #概率偏移，防止计算得到的数值过小
            for xi in range(len(X)):
                xiv = X[xi][d[xi]]    #取映射值
                p *= probXCY[y][xi][xiv]
            p *= probY[y]    # p{X1=x1|Y=y} * p{X2=x2|Y=y} *...* p{Xn=xn|Y=y} * p{y}
            probyCx.append(p)

        YofX.append(probyCx.index(max(probyCx)))  #max( p{Y=yi|X[...]=x[...]} )即取概率最大的那个分类yi为该数据的分类

        #分类记录
        print(d)
        print("predict_class：",YofX[-1])
        if(YofX[-1] != Cls[d[-1]]):
            diffNum += 1
            #print(probyCx)
            print("ture_class：",Cls[d[-1]])
        else:
            print("right")

    print("error_num：",diffNum,"\tdata_num:",datanum)
    print("accurate：",1 - diffNum/datanum)

    return YofX

#测试：
dS = readData('car.data.txt')
probXCY,probY = NBtrain(dS)
NBclassify(probXCY,probY,dS)