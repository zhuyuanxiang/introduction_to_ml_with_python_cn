# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C020306_decision_tree_ensemble.py
@Version    :   v0.1
@Time       :   2019-09-30 11:45
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec020306，P64
@Desc       :   监督学习算法。决策树集成。
"""

# Chap2 监督学习
import random
from math import log2
from math import sqrt

import matplotlib.pyplot as plt
import sklearn

import mglearn
from config import seed
from datasets.load_data import load_train_test_moons
from tools import plot_feature_importance
from tools import show_title


# 2.3. 监督学习算法
# 2.3.6 决策树集成
# 集成（ensemble）是合并多个机器学习模型来构建更加强大模型的方法。
# 1）随机森林：采用并行模式集成。
# 本质：许多决策树的集合，其中每棵树都与其他树略有不同。
# 思想：每棵树的预测都相对较好，但是对部分数据过拟合，通过对这些树的结果取平均值来降低过拟合。
# 优点：不需要反复调节参数，也不需要对数据进行缩放。
# 缺点：计算对于时间和空间的需要较大。
# RandomForestClassifier()
# n_estimators表示弱学习器的数目，数目越多，效果越好，学习时间越长
# max_features表示划分需要考虑数据集的特征数目，具体参考random_forest_cancer_dataset()的绘制图形
#   - 如果是1，那么划分时无法选择对哪个特征进行测试，就只能对随机选择某个特征搜索不同的阈值。可以降低过拟合。
#   - 如果是最大值，那么生成的树将会非常相似。
#   - 默认值。分类问题是sqrt(n_features)；回归问题是n_features。
# random_state：只是个随机数生成种子，保证每次生成一样的结果
# n_jobs表示可以使用的CPU的个数。-1表示全部使用。默认值是1。
def plot_decision_tree_and_forest():
    X_train, X_test, y_train, y_test = load_train_test_moons(n_samples=100, noise=0.25)

    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=5, random_state=2, n_jobs=3)
    forest.fit(X_train, y_train)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
        show_title(f"第{i}棵决策树")
        print(tree)
        ax.set_title(f'Tree{i}')
        mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
        pass
    mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
    axes[-1, -1].set_title('随机森林')
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.suptitle("图2-33：5棵随机化的决策树找到的决策边界+随机森林对预测概率取平均得到的决策边界")
    pass


def random_forest_max_feature_cancer():
    from sklearn.model_selection import train_test_split
    cancer = sklearn.datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=seed)

    # 训练随机森林
    print('=' * 20)
    from sklearn.ensemble import RandomForestClassifier
    max_feature_number = len(cancer.feature_names)
    for feature_number in [1,
                           round(sqrt(max_feature_number)),
                           round(log2(max_feature_number)),
                           round(max_feature_number * 0.3),
                           round(max_feature_number * 0.6),
                           max_feature_number]:
        # max_features不允许是浮点数
        forest = RandomForestClassifier(n_estimators=5, max_features=feature_number,
                                        random_state=seed, n_jobs=3)
        forest.fit(X_train, y_train)

        # 评价随机森林
        show_title("随机森林")
        print("最大特征数 = ", forest.max_features)
        print('训练集精度: {:.2f}'.format(forest.score(X_train, y_train)))
        print('测试集精度: {:.2f}'.format(forest.score(X_test, y_test)))
        pass
    pass


def random_forest_cancer_dataset():
    # 准备数据
    from sklearn.model_selection import train_test_split
    cancer = sklearn.datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=seed)

    # 训练随机森林
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, max_features=len(cancer.feature_names))
    forest.fit(X_train, y_train)

    # 评价随机森林
    show_title("随机森林")
    print("最大特征数 = ", forest.max_features)
    print('训练集精度: {:.2f}'.format(forest.score(X_train, y_train)))
    print('测试集精度: {:.2f}'.format(forest.score(X_test, y_test)))

    # 绘制随机森林的特征重要性
    plt.figure()
    plot_feature_importance(forest, cancer)
    plt.title("图2-34：拟合 Cancer 数据集得到的随机森林的特征重要性")

    # # 训练决策树
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(random_state=seed)
    tree.fit(X_train, y_train)

    # 评价决策树
    show_title("决策树")
    print('训练集精度: {:.2f}'.format(tree.score(X_train, y_train)))
    print('测试集精度: {:.2f}'.format(tree.score(X_test, y_test)))

    # # 绘制决策树的重要性
    plt.figure()
    plot_feature_importance(tree, cancer)
    plt.title('拟合 Cancer 数据集得到的决策树的特征重要性')

    # 使用随机森林的训练结果，从100个决策树中随机选择其中的7个决策树，绘制决策树的特征重要性
    for i in range(7):
        tree = random.choice(forest.estimators_)
        plt.figure()
        plot_feature_importance(tree, cancer)
        plt.title('从随机森林的100个决策树中随机选择的第{}个决策树的特征重要性'.format(i))
    pass


# 2) 梯度提升回归树（Gradient Boosted Decision Tree，GBDT）（梯度提升机）：采用串行模式集成。
# 通过合并多个决策树构建更加强大的模型。
# 既可以用于回归，也可以用于分类。
# 采用连续的方式构造树，每棵树都试图纠正前一棵树的错误。
# 默认情况下，GBDT没有随机化，用到了强预剪枝。
# 主要思想：合并许多简单的模型（即弱学习器）。每个模型只能对部分数据做出好的预测，添加的树越多，性能就越好。
# 相比随机森林，参数设置更加敏感，正确地设置参数后，可以得到较高的模型精度。
# 优点：不需要对数据缩放，适用于二元特征与连续特征同时存在的数据集。比随机森林消耗资源少。
# 缺点：受参数影响较大，训练时间较长。不适用于高维稀疏数据。
# 主要参数：
#   - n_estimators表示弱学习器的数目，数目越多，效果越好，学习时间越长，容易导致过拟合
#   - learning_rate：越低，就会需要更多的树构建具有同样复杂度的模型
#   - max_depth：降低每棵树的复杂度。
def plot_gbdt():
    from sklearn.model_selection import train_test_split
    cancer = sklearn.datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=seed)

    from sklearn.ensemble import GradientBoostingClassifier
    gbdt = GradientBoostingClassifier(random_state=seed)
    gbdt.fit(X_train, y_train)

    # 评价梯度提升决策树
    show_title("梯度提升决策树")
    print('训练集精度: {:.2f}'.format(gbdt.score(X_train, y_train)))
    print('测试集精度: {:.2f}'.format(gbdt.score(X_test, y_test)))

    plt.figure()
    plot_feature_importance(gbdt, cancer)
    plt.title("图2-34：拟合 Cancer 数据集得到的GBDT的特征重要性")


# 降低学习深度，加强预剪枝
def plot_gbdt_preprunning_max_depth():
    from sklearn.model_selection import train_test_split
    cancer = sklearn.datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                        random_state=seed)

    from sklearn.ensemble import GradientBoostingClassifier
    max_depth = 1
    gbdt = GradientBoostingClassifier(random_state=seed, max_depth=max_depth)
    gbdt.fit(X_train, y_train)

    # 评价「预剪枝梯度提升决策树」
    show_title("梯度提升决策树--降低学习深度，加强预剪枝")
    print("max depth = ", max_depth)
    print('训练集精度: {:.2f}'.format(gbdt.score(X_train, y_train)))
    print('测试集精度: {:.2f}'.format(gbdt.score(X_test, y_test)))

    plt.figure()
    plot_feature_importance(gbdt, cancer)
    plt.title("图2-34：拟合 Cancer 数据集得到的预剪枝（控制深度）GBDT的特征重要性")


# 降低学习率，加强预剪枝
# 学习率：表示每棵树纠正前一棵树的错误的强度。
#   - 高学习率表示每棵树可以做出强的修正，模型更加复杂。
def plot_gbdt_preprunning_learning_rate():
    from sklearn.model_selection import train_test_split
    cancer = sklearn.datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                        random_state=seed)

    from sklearn.ensemble import GradientBoostingClassifier
    learning_rate = 0.01
    gbdt = GradientBoostingClassifier(random_state=seed, learning_rate=learning_rate)
    gbdt.fit(X_train, y_train)

    # 评价「预剪枝梯度提升决策树」
    show_title("梯度提升决策树--降低学习率，加强预剪枝")
    print("learning rate = ", learning_rate)
    print('训练集精度: {:.2f}'.format(gbdt.score(X_train, y_train)))
    print('测试集精度: {:.2f}'.format(gbdt.score(X_test, y_test)))

    plt.figure()
    plot_feature_importance(gbdt, cancer)
    plt.title("图2-34：拟合 Cancer 数据集得到的预剪枝（控制学习率）GBDT的特征重要性")


if __name__ == "__main__":
    # 1）随机森林：采用并行模式集成。
    # 决策树与随机森林对比
    # plot_decision_tree_and_forest()

    # 随机森林最大特征数对学习的影响
    # random_forest_max_feature_cancer()

    # 随机森林与决策树学习 Cancer 数据集的特征重要性对比
    # random_forest_cancer_dataset()

    # 2) 梯度提升回归树（Gradient Boosted Decision Tree，GBDT）（梯度提升机）：采用串行模式集成。
    # plot_gbdt()

    # 降低深度，加强预剪枝
    plot_gbdt_preprunning_max_depth()

    # 降低学习率，加强预剪枝
    plot_gbdt_preprunning_learning_rate()

    from tools import beep_end
    from tools import show_figures

    beep_end()
    show_figures()
