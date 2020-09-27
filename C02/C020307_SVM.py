# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C020307_SVM.py
@Version    :   v0.1
@Time       :   2019-09-30 11:45
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec020307，P71
@Desc       :   监督学习算法。核支持向量机。
"""

# Chap2 监督学习
from config import *
from datasets.load_data import load_train_test_breast_cancer
from tools import *


# 1)线性模型与非线性特征


def load_two_nonlinear_blobs():
    """生成模拟数据"""
    from sklearn.datasets import make_blobs
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2  # 将生成的4类数据，变换成2类数据
    return X, y


def plot_datasets():
    X, y = load_two_nonlinear_blobs()

    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.legend()
    plt.suptitle("图2-36：二分类数据集（类别不是线性可分）")


def LinearSVC_blob_datasets():
    X, y = load_two_nonlinear_blobs()

    from sklearn.svm import LinearSVC
    linear_svm = LinearSVC().fit(X, y)

    mglearn.plots.plot_2d_separator(linear_svm, X)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.legend()
    plt.suptitle("图2-37：线性SVM给出的决策边界")


def plot_nonlinear_datasets():
    """为模拟数据增加一个新的特征，将线性不可分的数据变换到三维空间，变成线性可分"""
    X, y = load_two_nonlinear_blobs()

    # 添加一个新特征，新特征 = 第二个特征 ^ 2
    X_new = np.hstack([X, X[:, 1:] ** 2])
    figure = plt.figure()

    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(figure, elev=-152, azim=-26)
    mask = (y == 0)
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', cmap=mglearn.cm2, s=60, marker='^')
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set_zlabel('feature2 = feature1 ** 2')
    plt.suptitle("图2-38：将图2-37中的二维数据扩展成三维数据")


def LinearSVC_nonlinear_datasets():
    X, y = load_two_nonlinear_blobs()

    # 添加一个新特征，新特征 = 第二个特征 ^ 2
    X_new = np.hstack([X, X[:, 1:] ** 2])

    # 支持向量机 线性可分
    from sklearn.svm import LinearSVC
    linear_svm_3d = LinearSVC()
    linear_svm_3d.fit(X_new, y)
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

    # 3D图中显示线性决策边界
    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]

    from mpl_toolkits.mplot3d import Axes3D
    figure = plt.figure()
    ax = Axes3D(figure, elev=-152, azim=-26)
    ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
    mask = (y == 0)
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', cmap=mglearn.cm2, s=60, marker='^')
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set_zlabel('feature2 = feature1 ** 2')
    plt.suptitle("图2-39：线性SVM对扩展后的三维数据集给出的决策边界")


def plot_2d_LinearSVC_nonlinear_datasets():
    X, y = load_two_nonlinear_blobs()

    # 添加一个新特征，新特征 = 第二个特征 ^ 2
    X_new = np.hstack([X, X[:, 1:] ** 2])

    # 支持向量机 线性可分
    from sklearn.svm import LinearSVC
    linear_svm_3d = LinearSVC()
    linear_svm_3d.fit(X_new, y)

    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
    XX, YY = np.meshgrid(xx, yy)  # 生成网格点坐标矩阵
    ZZ = YY ** 2

    # 平面图中显示线性决策边界
    dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()], cmap=mglearn.cm2, alpha=0.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.legend()
    plt.suptitle("图2-40：将图2-39的三维决策边界投影到原始的二维平面上")
    pass


# 2) 核技巧（Kernel Trick）
# 核技巧：直接计算扩展特征表示中数据点之间的距离（即内积）。
# 常用核：多项式核；高斯核（径向基函数核）

# 3) 理解 SVM
# 在训练过程中，SVM学习每个训练数据点对于表示两个类别之间的决策边界的重要性。
# 只有一部分训练数据点对于定义决策边界很重要：位于类别之间边界上的点叫做支持向量。
# 分类决策面基于边界与支持向量之间的距离以及在训练过程中学到的支持向量的重要性来做出的。
# dual_coef_：保存SVC的支持向量的重要性
# 数据点之间的距离由高斯核给出k_{rbf}(x_1,x_2)=exp(-\gamma\|x_1-x_2\|^2)
def svc_2d_classification():
    # 两个数据中心的高斯数据
    X, y = mglearn.tools.make_handcrafted_dataset()

    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', C=10, gamma=0.1)
    svm.fit(X, y)
    mglearn.plots.plot_2d_separator(svm, X, eps=.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

    # 画出支持向量
    sv = svm.support_vectors_
    sv_labels = svm.dual_coef_.ravel() > 0
    mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.legend()
    plt.suptitle("图2-41：RBF核SVM给出的两个数据中心的高斯数据的决策边界和支持向量")
    pass


# 4) SVM调参
# 默认情况下：C=1，gamma=1/n_features
# 参数（gamma）：控制高斯核的宽度，决定了点与点之间“靠近”的距离。
#   - 值小，高斯核半径大，许多点都被认为“靠近”，决策边界变化慢，模型复杂度低；
#   - 值大，高斯核半径小，决策边界变化快，模型复杂度高。
# 参数（C）：正则化参数。
#   - 值小，模型受限，每个数据点的影响范围有限，决策边界是线性的，误分类的点对边界几乎没有影响；
#   - 值大，模型受每个数据点的影响变大，决策边界是非线性的，尽可能使每个点被正确分类。
def svc_difference_parameters():
    fig, axes = plt.subplots(3, 3, figsize=(20, 10))
    for ax, C in zip(axes, [-1, 0, 3]):
        for a, gamma in zip(ax, range(-1, 2)):
            mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
    axes[0, 0].legend(["Class 0", "Class 1", "支持向量 class 0", "支持向量 class 1"], ncol=4, loc=(.9, 1.2))
    plt.suptitle("图2-42：设置不同的 C 和 gamma 参数--两个数据中心的高斯数据对应的决策边界和支持向量")


def train_cancer_data_svc_over_fitting():
    # cancer 数据集的特征具有完全不同的数量级。对Kernel-SVM方法影响极大。
    # 注：实际情况是没有受到影响
    X_train, X_test, y_train, y_test = load_train_test_breast_cancer()

    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(X_train, y_train)
    print('=' * 20)
    print("-- Support Vector Classification Over-fitting --")
    print('Training set score: {:.2f}'.format(svc.score(X_train, y_train)))
    print('Test set score: {:.2f}'.format(svc.score(X_test, y_test)))

    plt.figure()
    plt.xlabel('Feature index')
    plt.ylabel('Feature magnitude')
    plt.yscale('log')
    plt.plot(X_train.min(axis=0), 'o', label='min')
    plt.plot(X_train.max(axis=0), 'o', label='max')
    plt.legend(loc=4)
    plt.suptitle("图2-43：Cancer 数据集的特征数据的取值范围--（注意y轴的对数坐标）")


# 5) 为 SVM　预处理数据
# 解决cancer数据集中特征数据的取值范围具有完全不同的数量级问题，可以先对数据进行“归一化”。
def normalize_cancer_data():
    """归一化cancer 数据集"""
    X_train, X_test, y_train, y_test = load_train_test_breast_cancer()

    min_on_training = X_train.min(axis=0)
    range_on_training = (X_train - min_on_training).max(axis=0)
    X_train_scaled = (X_train - min_on_training) / range_on_training
    X_test_scaled = (X_test - min_on_training) / range_on_training

    return X_train_scaled, X_test_scaled, y_train, y_test


def print_normalized_cancer_data():
    """输出归一化的数据集"""
    X_train_scaled, X_test_scaled, y_train, y_test = normalize_cancer_data()

    print('=' * 20)
    print("-- 归一化的数据集 --")
    print('Minimum for each feature\n{}'.format(X_train_scaled.min(axis=0)))
    print('Maximum for each feature\n{}'.format(X_train_scaled.max(axis=0)))


def plot_normalized_cancer_data():
    """显示归一化后的cancer数据集"""
    X_train_scaled, X_test_scaled, y_train, y_test = normalize_cancer_data()

    plt.xlabel('Feature index')
    plt.ylabel('Feature magnitude')
    plt.plot(X_train_scaled.min(axis=0), 'o', label='min')
    plt.plot(X_train_scaled.max(axis=0), 'o', label='max')
    plt.legend(loc=4)
    plt.suptitle("显示归一化后的cancer数据集")


def train_normalized_cancer_data_svc():
    X_train_scaled, X_test_scaled, y_train, y_test = normalize_cancer_data()

    from sklearn.svm import SVC
    svc = SVC(gamma='auto')
    svc.fit(X_train_scaled, y_train)
    print("-- Support Vecotr Classification --")
    print('=' * 20)
    print('default C', 'default gamma')
    print('Training set score: {:.3f}'.format(svc.score(X_train_scaled, y_train)))
    print('Test set score: {:.3f}'.format(svc.score(X_test_scaled, y_test)))

    # C 越小，考虑的是全局，模型也越简单；C 越大，考虑的是每个点的正确性，模型也越复杂
    print('=' * 20)
    for c_value in [1, 10, 100, 1000]:
        svc = SVC(C=c_value, gamma='auto')
        svc.fit(X_train_scaled, y_train)
        print('C=', c_value)
        print('Training set score: {:.3f}'.format(svc.score(X_train_scaled, y_train)))
        print('Test set score: {:.3f}'.format(svc.score(X_test_scaled, y_test)))
        print('-' * 20)
        pass

    # gamma越小，高斯核的范围越宽；gamma越大，高斯核的范围越窄
    # 过拟合会严重影响测试集的精确度
    print('=' * 20)
    for gamma in [0.1, 1, 10, 100]:
        svc = SVC(gamma=gamma)
        svc.fit(X_train_scaled, y_train)
        print('gamma=', gamma)
        print('Training set score: {:.3f}'.format(svc.score(X_train_scaled, y_train)))
        print('Test set score: {:.3f}'.format(svc.score(X_test_scaled, y_test)))
        print('-' * 20)

    # 两个最优的参数，结果不最优
    # 过拟合会严重影响测试集的精确度
    # 根据影响力调整的的参数，结果接近最优，重要的是防止过拟合，可以提高测试集的精确度
    print('=' * 20)
    for c_value, gamma in [(0.1, 10), (0.1, 20), (1, 1), (1, 10), (1, 20), (10, 1000)]:
        svc = SVC(C=c_value, gamma=gamma)
        svc.fit(X_train_scaled, y_train)
        print('C=', c_value, 'gamma=', gamma)
        print('Training set score: {:.3f}'.format(svc.score(X_train_scaled, y_train)))
        print('Test set score: {:.3f}'.format(svc.score(X_test_scaled, y_test)))
        print('-' * 20)
        pass
    pass


if __name__ == "__main__":
    # 生成模拟数据
    # plot_datasets()

    # 对模拟数据进行线性分类
    # LinearSVC_blob_datasets()

    # 为模拟数据增加一个新的特征，将线性不可分的数据变换到三维空间，变成线性可分
    # plot_nonlinear_datasets()

    # 图2-39：线性SVM对扩展后的三维数据集给出的决策边界
    # LinearSVC_nonlinear_datasets()

    # 图2-40：将图2-39的三维决策边界投影到原始的二维平面上
    # plot_2d_LinearSVC_nonlinear_datasets()

    # 图2-41：RBF核SVM给出的两个数据中心的高斯数据的决策边界和支持向量
    # svc_2d_classification()

    # 图2-42：设置不同的 C 和 gamma 参数--两个数据中心的高斯数据对应的决策边界和支持向量
    # svc_difference_parameters()

    # 图2-43：Cancer 数据集的特征数据的取值范围--（注意y轴的对数坐标）
    # train_cancer_data_svc_over_fitting()

    # 输出归一化的数据集
    # print_normalized_cancer_data()

    # 显示归一化后的cancer数据集
    # plot_normalized_cancer_data()

    # 学习归一化后的cancer数据集
    # train_normalized_cancer_data_svc()

    beep_end()
    show_figures()
