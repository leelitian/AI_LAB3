import numpy as np

# 映射属性值，方便代码处理
class_values = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}  # unacc, acc, good, vgood

# 特征值映射，共6个特征值，每个特征表示为X[i]，X[i][xiv]表示特征Xi的取值。
attributes = [
    {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3},  # buying: vhigh, high, med, low.
    {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3},  # maint: vhigh, high, med, low.
    {'2': 0, '3': 1, '4': 2, '5more': 3},  # doors: 2, 3, 4, 5more.
    {'2': 0, '4': 1, 'more': 2},  # persons: 2, 4, more.
    {'small': 0, 'med': 1, 'big': 2},  # lug_boot: small, med, big.
    {'low': 0, 'med': 1, 'high': 2}  # safety: low, med, high.
]


# 从文档中读取数据，每条数据转成列表的形式
def read_data(path):
    data_list = []
    with open(path, 'r') as f:
        data_set = f.readlines()

    for d in data_set:
        d = d[:-1]
        d = d.split(',')
        data_list.append(d)

    return data_list


# 将数据集划分为训练集和测试集
def split_data(data_list, test_scale):
    test_num = int(len(data_list) * test_scale)
    data_num = len(data_list)
    test_num = np.random.randint(0, data_num, test_num)  # 在总数据里取一部分做数据集，一部分做测试集

    train_data = []
    test_data = []
    for d in test_num:
        test_data.append(data_list[d])
    for d in range(data_num):
        if d not in test_num:
            train_data.append(data_list[d])

    print("data_set: {0}\ntrain_set: {1}\ntest_set: {2}".format(data_num, len(train_data), len(test_data)))
    return train_data, test_data
