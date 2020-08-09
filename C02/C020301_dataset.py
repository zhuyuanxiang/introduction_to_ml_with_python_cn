# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C020301_dataset.py
@Version    :   v0.1
@Time       :   2019-09-19 17:52
@License    :   (C)Copyright 2019-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec020301，P24
@Desc       :   监督学习算法。创建样本数据集并展示数据集的效果图
"""

# 2.3. 监督学习算法
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import sklearn

from mglearn import datasets


# 2.3.1. 一些样本数据集
# 生成forge数据集，是一个二分类数据集，有两个特征
def create_forge():
    X, y = datasets.make_forge()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(['Class 0', 'Class 1'], loc = 4)
    plt.xlabel('First feature')
    plt.ylabel('Second feature')
    print('X.shape: {}'.format(X.shape))
    print('y.shape: {}'.format(y.shape))
    plt.plot(X[:, 0], X[:, 1], 'x')
    plt.suptitle("图2-2：forge数据集的散点图")


# 生成wave数据集，只有一个输入特征和一个连续的目标变量（响应）
def create_wave():
    X, y = datasets.make_wave()
    mglearn.discrete_scatter(X, y)
    # plt.figure()
    # plt.plot(X, y, '^')
    plt.ylim(-3, 3)
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.suptitle("图2-3：wave数据集的图像，\n"
                 "x轴表示特征，y轴表示回归目标")
    print('X.shape: {}'.format(X.shape))
    print('y.shape: {}'.format(y.shape))


# breast cancer 威斯康星州乳腺癌分类数据集，恶性（malignant=0）和良性（benign=1）
def load_breast_cancer():
    breast_cancer = sklearn.datasets.load_breast_cancer()
    print('cancer.keys():\n\t{}'.format(breast_cancer.keys()))
    print('Shape of cancer data: {}'.format(breast_cancer.data.shape))
    # target_names与target之间的关系是默认设定的
    print('Sample counts per class:\n\t{}'.format(
            {n: v for n, v in zip(breast_cancer.target_names, np.bincount(breast_cancer.target))}))

    print('Sample counts per class:\n\t{}'.format(
            list(zip(list(breast_cancer.target_names),
                     list(np.bincount(breast_cancer.target))))))

    print('Feature names:\n{}'.format(breast_cancer.feature_names))
    print('Shape of cancer feature data: {}'.format(breast_cancer.feature_names.data.shape))
    return breast_cancer


# boston 波士顿房价回归数据集
def load_boston():
    boston = sklearn.datasets.load_boston()
    print('Data shape: {}'.format(boston.data.shape))

    # 波士顿房价扩展后的回归数据集，输入特征包括13个测量结果和这些特征之间的乘积（交互项）
    # 特征工程（Feature Engineering）：包含了导出特征的方法。
    X, y = mglearn.datasets.load_extended_boston()
    print('X.shape: {}'.format(X.shape))


if __name__ == "__main__":
    # 2.3.1. 一些样本数据集
    # 生成forge数据集，是一个二分类数据集，有两个特征
    create_forge()

    # 生成wave数据集，只有一个输入特征和一个连续的目标变量（响应）
    create_wave()

    # breast cancer 威斯康星州乳腺癌分类数据集，恶性（malignant=0）和良性（benign=1）
    cancer = load_breast_cancer()

    # boston 波士顿房价回归数据集
    load_boston()

    import tools
    tools.beep_end()
    tools.show_figures()