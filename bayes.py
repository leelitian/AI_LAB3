import numpy as np
from utils import class_values
from utils import attributes
import utils


def train(train_data):
    """
    :param train_data: 训练集
    :return: p( xi = xiv | y = yi ) , p{ y = yi }
    """

    data_num = len(train_data)

    num_matrix = np.zeros((len(class_values), len(attributes), 4))  # number( xi = xiv | y = yi )
    prob_matrix = np.zeros((len(class_values), len(attributes), 4))  # p( xi = xiv | y = yi )

    for row in train_data:
        yi = class_values[row[-1]]  # 取分类的映射值
        for xi in range(len(row) - 1):
            xiv = attributes[xi][row[xi]]  # 取每个特征的映射值
            num_matrix[yi][xi][xiv] += 1  # 计数

    num_yi = []  # 分类为yi的数量
    prob_yi = []  # 分类为yi的概率

    for yi in class_values.values():
        num_yi.append(np.sum(num_matrix[yi][0]))
        # prob_yi.append(num_yi[yi] / data_num)  # 优化前
        prob_yi.append((num_yi[yi] + 1) / (data_num + len(class_values)))  # 做拉普拉斯平滑

        for xi in range(len(attributes)):
            s = len(attributes[xi])  # 特征attribute的值的个数
            for xiv in attributes[xi].values():
                # prob_matrix[yi][xi][xiv] = num_matrix[yi][xi][xiv] / num_yi[yi]  # 优化前
                prob_matrix[yi][xi][xiv] = (num_matrix[yi][xi][xiv] + 1) / (num_yi[yi] + s)  # 做拉普拉斯平滑

    return prob_matrix, prob_yi


# 测试模型
def test(prob_matrix, prob_yi, test_data):
    """
    :param prob_matrix: 概率矩阵 p( xi = xiv | y = yi )
    :param prob_yi:     属性值为yi的概率p{ y = yi }
    :param test_data:   测试集数据
    :return:            打印结果
    """

    data_num = len(test_data)
    error = 0  # 记录分类结果与实际不同的数量

    for row in test_data:
        prob = []  # p( y = yi | X.. = x.. )

        for yi in class_values.values():
            p = 10 ** 5  # 概率偏移，防止计算得到的数值过小
            for xi in range(len(attributes)):
                xiv = attributes[xi][row[xi]]
                p = p * prob_matrix[yi][xi][xiv]

            p = p * prob_yi[yi]  # p{X1=x1|Y=y} * p{X2=x2|Y=y} *...* p{Xn=xn|Y=y} * p{y}
            prob.append(p)

        result = prob.index(max(prob))  # 取最大的作为结果yi

        if result != class_values[row[-1]]:
            error += 1

    print("error_num：", error)
    print("accuracy：{}\n".format(1 - error / data_num))


def main():
    print("--------- Naive Bayes ---------")
    data_set = utils.read_data('dataset.txt')
    train_data, test_data = utils.split_data(data_set, 0.3)

    prob_matrix, prob_yi = train(train_data)
    test(prob_matrix, prob_yi, test_data)
