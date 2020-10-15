# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C020302_K_neighbors.py
@Version    :   v0.1
@Time       :   2019-09-18 14:18
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec020302，P28
@Desc       :   监督学习算法。K近邻
"""

# 2.3. 监督学习算法
from config import *
from datasets.load_data import *
from tools import *


# 2.3.2. k-NN（k近邻）
# 1) k近邻分类
def compare_NeighborsNumber():
    # 左上角的数据点和n_neighbors=1时不同
    # 测试数据点并没有正确的值，是随意确定的几个值，观察它们如何受n_neighbor的影响
    for n_neighbors in [1, 3, 5, 7]:
        plt.figure()
        mglearn.plots.plot_knn_classification(n_neighbors=n_neighbors)
        plt.suptitle(f"图2-5：{n_neighbors}近邻模型对forge数据集的预测结果")
        pass


# 训练K近邻分类模型
def fit_KNeighborsClassifier():
    # 准备forge数据集(二分类问题)，数据越多越准确
    # 会掩蔽 4 条数据
    # X, y = datasets.make_forge()
    X, y = make_my_forge(n_samples=30)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    from mglearn import discrete_scatter
    discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers=['^'])
    discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers=['*'])
    # plt.scatter(X_train[:, 0], X_train[:, 1], marker='^')
    # plt.scatter(X_test[:, 0], X_test[:, 1], marker='v')

    from sklearn.neighbors import KNeighborsClassifier
    for n_neighbors in [1, 3]:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        show_title(f'邻居个数={n_neighbors}')
        print('测试集预测: {}'.format(clf.predict(X_test)))
        print('测试集精度: {:.2f}'.format(clf.score(X_test, y_test)))
    print("测试集误差：{}".format((clf.predict(X_test) - y_test).sum()))


# 2) 分析 KNeighborsClassifier() 函数
# 不同n_neighbors值的K近邻模型的决策边界
def analysis_KNeighborsClassifier():
    # 准备forget数据集，数据越多越准确
    X, y = make_my_forge(n_samples=30)
    # 邻居数越少，决策边界越受每一个数据的特性影响；
    # 邻居数越大，决策边界越受所有数据的平均特性影响，即决策边界会越平滑。
    from sklearn.neighbors import KNeighborsClassifier
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    plt.suptitle("图2-6：不同邻居个数的K近邻模型的决策边界")
    for n_neighbors, ax in zip([1, 3, 9], axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title(f'{n_neighbors} 个邻居')
        ax.set_xlabel('特征 0')
        ax.set_ylabel('特征 1')

    # 不同的位置显示图例：
    axes[0].legend()
    # axes[0].legend(loc = 0)
    # axes[0].legend(loc = 1)
    # axes[0].legend(loc = 2)
    # axes[0].legend(loc = 3)
    # axes[0].legend(loc = 4)
    # axes[0].legend(loc = 5)
    # axes[0].legend(loc = 6)
    # axes[0].legend(loc = 7)
    # axes[0].legend(loc = 8)
    # axes[0].legend(loc = 9)
    # axes[0].legend(loc = 10)
    # axes[0].legend(loc = 11)  # 没有这个位置


# 以n_neighbors为自变量，对比训练集精度和测试集精度
def analysis_ModelComplexity():
    X_train, X_test, y_train, y_test = load_train_test_breast_cancer()
    training_accuracy, test_accuracy = [], []
    neighbors_settings = range(1, 21)

    # 邻居个数过少时，模型欠拟合，测试数据集的精确度较差，模型泛化能力弱；
    # 邻居个数增加时，模型泛化能力增加，测试数据集的精确度也增加，
    # 当达到顶点后，模型泛化能力开始下降，测试数据的精确度也开始下降，模型过拟合。
    from sklearn.neighbors import KNeighborsClassifier
    for n_neighbors in neighbors_settings:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
        training_accuracy.append(clf.score(X_train, y_train))  # 记录训练精度
        test_accuracy.append((clf.score(X_test, y_test)))  # 记录测试精度

    plt.plot(neighbors_settings, training_accuracy, label='训练精度')
    plt.plot(neighbors_settings, test_accuracy, label='测试精度')
    plt.xlabel('邻居个数')
    plt.ylabel('精度')
    plt.legend()
    plt.suptitle("图2-7：以n_neighbors为自变量，对比训练集精度和测试集精度")


# 3) K近邻回归
def compare_KNeighborsRegressor():
    # 左边第1个数据预测值变化较大
    for n_neighbors in [1, 3, 5, 7]:
        mglearn.plots.plot_knn_regression(n_neighbors=n_neighbors)
        plt.suptitle(f"图2-8：{n_neighbors}近邻回归对wave数据集的预测结果")
        pass


def fit_KNeighborsRegressor():
    X_train, X_test, y_train, y_test = load_train_test_wave(100)
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)
    from sklearn.neighbors import KNeighborsRegressor
    for n_neighbors in [1, 3, 5, 7, 9, 11]:
        regress = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_train)
        show_title(f'邻居个数: {n_neighbors}')
        print(f'测试集的预测结果:\n{regress.predict(X_test)}')
        # R^2分数，叫做决定系数，是回归模型预测的优度度量。
        '''The coefficient R^2 is defined as (1 - u/v), 
        where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() 
        and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). 
        A constant model that always predicts the expected value of y, 
        disregarding the input features, would get a R^2 score of 0.0.'''
        print('测试集的分类 R^2: {:.2f}'.format(regress.score(X_test, y_test)))
        print('-----')


# 4) 分析 KNeighborsRegressor() 函数
def analysis_KNeighborsRegressor():
    X_train, X_test, y_train, y_test = load_train_test_wave(n_samples=140)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    print(line.data.shape)

    from sklearn.neighbors import KNeighborsRegressor
    for n_neighbors, ax in zip([1, 3, 5, 7, 9, 11], axes.reshape(6)):
        reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_train)
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=4)
        ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=4)
        ax.set_title('{} 个邻居： 训练得分{:.2f} 测试得分{:.2f}'.format(
                n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
        ax.set_xlabel('特征')
        ax.set_ylabel('目标')

    axes[0, 0].legend(['模型预测', '训练集', '测试集'], loc='best')
    plt.suptitle("图2-10：不同邻居个数的K近邻回归的预测结果对比\n邻居个数越大，回归曲线越平滑")


if __name__ == "__main__":
    # 1) k近邻分类对forge数据集的预测效果
    # compare_NeighborsNumber()

    # 训练K近邻分类模型
    # fit_KNeighborsClassifier()

    # 2) 分析 KNeighborsClassifier() 函数
    # 不同n_neighbors值的K近邻模型的决策边界
    # analysis_KNeighborsClassifier()

    # 以n_neighbors为自变量，对比训练集精度和测试集精度
    # analysis_ModelComplexity()

    # 3) K近邻回归
    # K近邻回归效果比较
    # 不同n_neighbors值的K近邻回归模型对wave数据集的预测结果
    # compare_KNeighborsRegressor()

    # K近邻回归的训练
    fit_KNeighborsRegressor()

    # 4) 分析 KNeighborsRegressor() 函数
    # analysis_KNeighborsRegressor()

    beep_end()
    show_figures()
