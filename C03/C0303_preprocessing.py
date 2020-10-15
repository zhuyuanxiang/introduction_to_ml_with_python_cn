# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C0303_preprocessing.py
@Version    :   v0.1
@Time       :   2019-10-06 08:34
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec0303，P101
@Desc       :   无监督学习算法。数据预处理和缩放。
"""

from tools import *


# 3.3. 预处理和缩放
# StandardScaler    ：确保每个特征的平均值为0，方差为1，使所有特征都位于同一量级。
# RobustScaler      ：确保每个特征的统计属于都位于同一范围，中位数为0，四分位数为1？，从而忽略异常值
# MinMaxScaler      ：确保所有特征都位于0到1之间
# Normalizer        ：归一化。对每个数据点都进行缩放，使得特征向量的欧氏长度为1。
#                       即将数据点都投向到半径为1的圆上。
#                       因此，只关注数据的方向（或角度），不关注数据的长度。
def data_scale():
    mglearn.plots.plot_scaling()
    plt.suptitle("图3-1 对数据集缩放和预处理的各种方法\n"
                 "|StandardScaler(特征位于同一量级,均值0,方差1)                     |    Normalizer(归一化。特征向量欧氏长度为1)|\n"
                 "|RobustScaler(特征统计位于同一范围,中位数0,四分位数1,忽略异常值)    |            MinMaxScaler(特征位于0到1之间)|")


# 3.3.2. 应用数据变换
# 数据变换尺度比例是基于训练数据集完成的，因此测试数据集变换后区间不在[0,1]之间
def min_max_scaler():
    # 导入数据
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=seed)

    # 导入预处理器
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # 训练预处理器
    scaler.fit(X_train)

    # 变换数据
    X_train_scaled = scaler.transform(X_train)

    print('=' * 20)
    print("训练数据集变换前的形状：", X_train.shape)
    print("训练数据集变换后的形状：", X_train_scaled.shape)
    print('-' * 20)
    print('训练数据集变换前每一个特征的最小值:\n{}'.format(X_train.min(axis=0)))
    print('-' * 20)
    print('训练数据集变换前每一个特征的最大值:\n{}'.format(X_train.max(axis=0)))
    print('-' * 20)
    print('训练数据集变换后每一个特征的最小值:\n{}'.format(X_train_scaled.min(axis=0)))
    print('-' * 20)
    print('训练数据集变换后每一个特征的最大值:\n{}'.format(X_train_scaled.max(axis=0)))

    X_test_scaled = scaler.transform(X_test)
    print('=' * 20)
    print("测试数据集变换前的形状：", X_test.shape)
    print("测试数据集变换后的形状：", X_test_scaled.shape)
    print('-' * 20)
    print('测试数据集变换前每一个特征的最小值:\n{}'.format(X_test.min(axis=0)))
    print('-' * 20)
    print('测试数据集变换前每一个特征的最大值:\n{}'.format(X_test.max(axis=0)))
    print('-' * 20)
    print('测试数据集变换后每一个特征的最小值:\n{}'.format(X_test_scaled.min(axis=0)))
    print('-' * 20)
    print('测试数据集变换后每一个特征的最大值:\n{}'.format(X_test_scaled.max(axis=0)))


def stand_scaler():
    # 导入数据
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=seed)

    # 导入预处理器
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # 训练预处理器
    scaler.fit(X_train)

    # 变换数据
    X_train_scaled = scaler.transform(X_train)

    print('=' * 20)
    print("训练数据集变换前的形状：", X_train.shape)
    print("训练数据集变换后的形状：", X_train_scaled.shape)
    print('-' * 20)
    print('训练数据集变换前每一个特征的最小值:\n{}'.format(X_train.min(axis=0)))
    print('训练数据集变换后每一个特征的最小值:\n{}'.format(X_train_scaled.min(axis=0)))
    print('-' * 20)
    print('训练数据集变换前每一个特征的最大值:\n{}'.format(X_train.max(axis=0)))
    print('训练数据集变换后每一个特征的最大值:\n{}'.format(X_train_scaled.max(axis=0)))
    print('-' * 20)
    print('-' * 20)
    print(f"训练数据集变换前的均值：\n{X_train.mean()}")
    print(f"训练数据集变换后的均值：\n{X_train_scaled.mean()}")
    print('-' * 20)
    print(f"训练数据集变换前的标准差：\n{X_train.std()}")
    print(f"训练数据集变换后的标准差：\n{X_train_scaled.std()}")

    X_test_scaled = scaler.transform(X_test)
    print('=' * 20)
    print("测试数据集变换前的形状：", X_test.shape)
    print("测试数据集变换后的形状：", X_test_scaled.shape)
    print('-' * 20)
    print('测试数据集变换前每一个特征的最小值:\n{}'.format(X_test.min(axis=0)))
    print('-' * 20)
    print('测试数据集变换前每一个特征的最大值:\n{}'.format(X_test.max(axis=0)))
    print('-' * 20)
    print('测试数据集变换后每一个特征的最小值:\n{}'.format(X_test_scaled.min(axis=0)))
    print('-' * 20)
    print('测试数据集变换后每一个特征的最大值:\n{}'.format(X_test_scaled.max(axis=0)))


# 3.3.3. 对训练数据和测试数据进行相同的缩放
def min_max_blob_data():
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    X, _ = make_blobs(n_samples=50, centers=5, random_state=seed, cluster_std=2)
    X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    # 绘制原始的训练集和测试集
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c='blue', s=60, label="训练数据集")
    axes[0].scatter(X_test[:, 0], X_test[:, 1], c='red', s=60, label="测试数据集", marker='^')
    axes[0].set_title('原始数据')
    axes[0].legend(loc='upper left')

    # 绘制MinMaxScaler()缩放的训练集和测试集
    # 导入预处理器
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # 使用X_train训练，对所有数据同时缩放
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c='blue', s=60, label='训练数据集')
    axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c='red', s=60, label='测试数据集', marker='^')
    axes[1].set_title('正确缩放后的数据')

    # 对X_test训练，对测试数据单独缩放（这样的缩放方式是错误的），破坏了训练数据与测试数据的相关性
    test_scaler = MinMaxScaler()
    test_scaler.fit(X_test)
    X_test_scaled_badly = test_scaler.transform(X_test)
    axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c='blue', s=60, label='训练数据集')
    axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], c='red', s=60, label='测试数据集',
                    marker='^')
    axes[2].set_title('错误缩放后的数据')

    plt.suptitle("图3-2：原始数据（左），同时缩放的数据（中），分别缩放的数据（右）")


# 3.3.4. 预处理对监督学习的作用
# SVM对数据缩放比较敏感
def preprocess_data_svm():
    # 导入数据
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=seed)

    from sklearn.svm import SVC
    svm = SVC(C=100, gamma='auto')
    svm.fit(X_train, y_train)
    print('=' * 20)
    print("-- 没有预处理的数据通过SVC学习 --")
    print('测试集的精确度: {:.2f}'.format(svm.score(X_test, y_test)))

    # 使用0-1缩放进行预处理的数据通过SVC学习
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm.fit(X_train_scaled, y_train)
    print('-' * 20)
    print("-- 使用0-1缩放进行预处理的数据通过SVC学习 --")
    print('缩放后测试集的精确度: {:.2f}'.format(svm.score(X_test_scaled, y_test)))

    # 使用0均值，1方差，缩放进行预处理的数据
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm.fit(X_train_scaled, y_train)
    print('-' * 20)
    print("-- 使用0均值，1方差，缩放进行预处理的数据通过SVC学习 --")
    print('缩放后测试集的精确度 {:.2f}'.format(svm.score(X_test_scaled, y_test)))


def main():
    # 图3-1 对数据集缩放和预处理的各种方法
    # data_scale()
    # 使用0-1缩放进行预处理的数据
    # min_max_scaler()
    stand_scaler()
    # 图3-2：原始数据（左），同时缩放的数据（中），分别缩放的数据（右）
    # min_max_blob_data()
    # SVM对数据缩放比较敏感
    # preprocess_data_svm()
    pass


if __name__ == "__main__":
    main()
    beep_end()
    show_figures()
