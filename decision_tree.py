from math import log
from utils import class_values
from utils import attributes
import utils


def count_each_class(data_set):
    """
    :param data_set: 数据集
    :return:         数据集根据属性值class_value分为几类
    """
    num_each_class = [0] * len(class_values)  # 列表初始化

    for d in data_set:
        num_each_class[class_values[d[-1]]] += 1

    return num_each_class


def entropy(data_set):
    """
    :param data_set:    数据集
    :return:            数据集的熵
    """
    num_each_class = count_each_class(data_set)
    data_num = len(data_set)
    ent = 0

    for num in num_each_class:
        scale = num / data_num
        if scale != 0:
            ent = ent - scale * log(scale, 2)  # 熵计算公式

    return ent


def split_data(data_set, xi):
    """
    :param data_set: 原数据集
    :param xi:      用以划分的属性xi
    :return:        [[xi0对应的数据集], [xi1], [xi2] ...]
    """
    data_subsets = [[] for _ in range(len(attributes[xi]))]  # 子数据集列表初始化

    for d in data_set:
        data_subsets[attributes[xi][d[xi]]].append(d)

    return data_subsets


def calculate_gain(data_set, xi):  # xi：属性xi
    """
    :param data_set: 原数据集
    :param xi:       用来划分的属性xi
    :return:         划分后的信息增益
    """
    result_ent = 0
    origin_ent = entropy(data_set)  # 熵
    data_subset = split_data(data_set, xi)

    for data_xiv in data_subset:
        if data_xiv:
            result_ent += len(data_xiv) / len(data_set) * entropy(data_xiv)  # 带入计算公式

    # 原来的熵 - 现在的熵
    return origin_ent - result_ent


def get_max_gain(data_set, used_x=[]):
    """
    :param data_set:
    :param used_x:  已经使用过的attribute
    :return:        获得最大的信息增益值和对应的特征序号
    """
    gains = []
    for xi in range(len(attributes)):
        if xi not in used_x:
            gains.append(calculate_gain(data_set, xi))
        else:
            gains.append(0)

    max_gain = max(gains)  # 获取最大收益
    max_gain_index = gains.index(max_gain)  # 获取最大收益索引

    return max_gain_index, max_gain


# 以字典的结构构建决策树
def create_tree(data_set, r, used_x=[]):
    if len(data_set) == 0:
        return {}  # 空树

    tree = {}
    num_class = count_each_class(data_set)
    mode = num_class.index(max(num_class))

    tree['class'] = mode  # 该树各分类中数据最多的类，记为该根节点的分类
    max_gain_index, max_gain = get_max_gain(data_set, used_x)
    # print("max gain:", max_gain)

    if len(used_x) == len(attributes) or num_class[mode] == len(data_set) or max_gain < r:
        tree['feature'] = -1  # 不在继续分支，即为叶节点
        return tree

    else:
        tree['feature'] = max_gain_index  # 记录该根节点用于划分的特征
        data_subsets = split_data(data_set, max_gain_index)  # 用该特征的值划分子集，用于构建子树

        for xiv in range(len(attributes[max_gain_index])):
            data_set_xiv = data_subsets[xiv]
            new_used_x = used_x.copy()
            new_used_x.append(max_gain_index)
            tree[xiv] = create_tree(data_set_xiv, r, new_used_x)  # 递归构建一棵树

    return tree


def classify(tree, data):
    xi = tree['feature']  # 根节点用于划分子树的特征
    if xi == -1:
        return tree['class']

    subtree = tree[attributes[xi][data[xi]]]
    if subtree == {}:  # 节点没有该子树
        return tree['class']  # 以该节点的分类作为数据的分类
    return classify(subtree, data)  # 否则遍历子树


def test(tree, test_data):
    """
    :param tree:        构建出来的决策树
    :param test_data:   测试集
    :return:            打印测试结果
    """
    error = 0

    for d in test_data:
        c = classify(tree, d)
        if c != class_values[d[-1]]:
            error = error + 1

    test_num = len(test_data)
    print("error_num：", error)
    print("accuracy：{}\n".format(1 - error / test_num))


def main():
    print("--------- Decision Tree ---------")
    data_set = utils.read_data('dataset.txt')
    train_data, test_data = utils.split_data(data_set, 0.3)
    tree = create_tree(train_data, 0.2)
    test(tree, test_data)
