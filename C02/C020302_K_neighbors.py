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
import matplotlib.pyplot as plt
import numpy as np
import mglearn
import sklearn

np.set_printoptions(precision = 3, suppress = True, threshold = np.inf)


# 2.3.2. k-NN（k近邻）
# 1) k近邻分类
def compare_NeighborsNumber():
    # 左上角的数据点和n_neighbors=1时不同
    for n_neighbors in [1, 3, 5, 7]:
        plt.figure()
        mglearn.plots.plot_knn_classification(n_neighbors = n_neighbors)
        plt.suptitle("图2-5：{}近邻模型对forge数据集的预测结果".format(n_neighbors))
        pass

# 训练K近邻分类模型
def fit_KNeighborsClassifier():
    # 准备forget数据集，数据越多越准确
    X, y = mglearn.datasets.make_forge()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors = 3)
    clf.fit(X_train, y_train)
    print('Test set predictions: {}'.format(clf.predict(X_test)))
    print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))


# 2) 分析 KNeighborsClassifier() 函数
# 不同n_neighbors值的K近邻模型的决策边界
def analysis_KNeighborsClassifier():
    # 准备forget数据集，数据越多越准确
    from sklearn.model_selection import train_test_split
    X, y = mglearn.datasets.make_forge()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    # 邻居数越少，决策边界越受每一个数据的特性影响；
    # 邻居数越大，决策边界越受所有数据的平均特性影响，即决策边界会越平滑。
    from sklearn.neighbors import KNeighborsClassifier
    fig, axes = plt.subplots(1, 3, figsize = (10, 3))
    plt.suptitle("图2-6：不同n_neighbors值的K近邻模型的决策边界")
    for n_neighbors, ax in zip([1, 3, 9], axes):
        clf = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill = True, eps = 0.5, ax = ax, alpha = .4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax = ax)
        ax.set_title('{} neighbor(s)'.format(n_neighbors))
        ax.set_xlabel('feature 0')
        ax.set_ylabel('feature 1')

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
    from sklearn.model_selection import train_test_split
    cancer = sklearn.datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, stratify = cancer.target, random_state = 66)
    training_accuracy = []
    test_accuracy = []
    neighbors_settings = range(1, 11)

    # 邻居个数过少时，模型欠拟合，测试数据集的精确度较差，模型泛化能力弱；
    # 邻居个数增加时，模型泛化能力增加，测试数据集的精确度也增加，
    # 当达到顶点后，模型泛化能力开始下降，测试数据的精确度也开始下降，模型过拟合。
    from sklearn.neighbors import KNeighborsClassifier
    for n_neighbors in neighbors_settings:
        clf = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X_train, y_train)
        training_accuracy.append(clf.score(X_train, y_train))  # 记录训练精度
        test_accuracy.append((clf.score(X_test, y_test)))  # 记录测试精度

    plt.plot(neighbors_settings, training_accuracy, label = 'training accuracy')
    plt.plot(neighbors_settings, test_accuracy, label = 'test accuracy')
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.suptitle("图2-7：以n_neighbors为自变量，对比训练集精度和测试集精度")



# 3) K近邻回归
def compare_KNeighborsRegressor():
    # 左边第1个数据预测值变化较大
    for n_neighbors in [1, 3, 5, 7]:
        mglearn.plots.plot_knn_regression(n_neighbors = n_neighbors)
        plt.suptitle("图2-8：{}近邻回归对wave数据集的预测结果".format(n_neighbors))
        pass


def fit_KNeighborsRegressor():
    from sklearn.model_selection import train_test_split
    X, y = mglearn.datasets.make_wave(n_samples = 40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    from sklearn.neighbors import KNeighborsRegressor
    for n_neighbors in [1, 3, 5, 7]:
        regress = KNeighborsRegressor(n_neighbors = n_neighbors).fit(X_train, y_train)
        print('n_neighbors: {}'.format(n_neighbors))
        print('Test set predictions:\n{}'.format(regress.predict(X_test)))
        # R^2分数，叫做决定系数，是回归模型预测的优度度量。
        '''The coefficient R^2 is defined as (1 - u/v), 
        where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() 
        and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). 
        A constant model that always predicts the expected value of y, 
        disregarding the input features, would get a R^2 score of 0.0.'''
        print('Test set R^2: {:.2f}'.format(regress.score(X_test, y_test)))
        print('-----')


# 4) 分析 KNeighborsRegressor() 函数
def analysis_KNeighborsRegressor():
    from sklearn.model_selection import train_test_split
    X, y = mglearn.datasets.make_wave(n_samples = 40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    fig, axes = plt.subplots(2, 3, figsize = (15, 8))
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    print(line.data.shape)

    from sklearn.neighbors import KNeighborsRegressor
    for n_neighbors, ax in zip([1, 3, 5, 7, 9, 11], axes.reshape(6)):
        reg = KNeighborsRegressor(n_neighbors = n_neighbors).fit(X_train, y_train)
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, '^', c = mglearn.cm2(0), markersize = 8)
        ax.plot(X_test, y_test, 'v', c = mglearn.cm2(1), markersize = 8)
        ax.set_title('{} neighbor(s)\n train score: {:.2f} test score: {:.2f}'.format(
                n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
        ax.set_xlabel('Feature')
        ax.set_ylabel('Target')

    axes[0, 0].legend(['Model predictions', 'Training data/target',
                       'Test data/target'], loc = 'best')
    plt.suptitle("图2-10：不同n_neighbors值的K近邻回归的预测结果对比")


if __name__ == "__main__":
    # 1) k近邻分类对forge数据集的预测效果
    compare_NeighborsNumber()

    # 训练K近邻分类模型
    fit_KNeighborsClassifier()

    # 2) 分析 KNeighborsClassifier() 函数
    # 不同n_neighbors值的K近邻模型的决策边界
    analysis_KNeighborsClassifier()

    # 以n_neighbors为自变量，对比训练集精度和测试集精度
    analysis_ModelComplexity()

    # 3) K近邻回归
    # K近邻回归效果比较
    # 不同n_neighbors值的K近邻回归模型对wave数据集的预测结果
    compare_KNeighborsRegressor()

    # K近邻回归的训练
    fit_KNeighborsRegressor()

    # 4) 分析 KNeighborsRegressor() 函数
    analysis_KNeighborsRegressor()

    import winsound
    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
