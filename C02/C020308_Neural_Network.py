# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C020308_Neural_Network.py
@Version    :   v0.1
@Time       :   2019-10-05 10:13
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec020308，P80
@Desc       :   监督学习算法。神经网络（深度学习）
"""
# Chap2 监督学习
import config
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import mglearn
import sklearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from tools import beep_end, show_figures


# 2.3.8. 神经网络（深度学习）
# 图2-44：Logistic回归的可视化


def logistic_graph():
    """Logistic回归的可视化"""
    # 不在IPython中无法显示，改用PDF输出
    regression_graph = mglearn.plots.plot_logistic_regression_graph()
    graph = graphviz.Source(regression_graph)
    graph.render('tree')
    graph.view()


# 图2-45：单隐层的多层感知机可视化
def single_hidden_layer_graph():
    """单隐层的多层感知机可视化"""
    # 不在IPython中无法显示，改用PDF输出
    layer_graph = mglearn.plots.plot_single_hidden_layer_graph()
    graph = graphviz.Source(layer_graph)
    graph.render('tree')
    graph.view()


# 图2-47：双隐层的多层感知机可视化
def two_hidden_layer_graph():
    """双隐层的多层感知机可视化"""
    # 不在IPython中无法显示，改用PDF输出
    layer_graph = mglearn.plots.plot_two_hidden_layer_graph()
    graph = graphviz.Source(layer_graph)
    graph.render('tree')
    graph.view()


def two_activate_function():
    """两种激活函数的对比图"""
    # NN 的激活函数
    line = np.linspace(-3, 3, 100)
    # 双曲正切激活函数(捕捉更多数据的细节)
    plt.plot(line, np.tanh(line), label = 'tanh')
    # 校正线性激活函数(捕捉更多数据的趋势)
    plt.plot(line, np.maximum(line, 0), label = 'relu')
    plt.legend(loc = 'best')
    plt.xlabel('x')
    plt.ylabel('relu(x), tanh(x)')
    plt.suptitle("图2-46：两种激活函数的对比图\n"
                 "双曲正切激活函数 和 校正线性激活函数")


# 2) 神经网络调参
def tune_neural_network_parameter():
    """神经网络的参数调整"""
    # 通过对比，得出结论。
    # 单元数越多，决策边界越趋向曲线；
    # 隐层数越多，决策边界的弯曲越多；
    # 激活函数为relu时，决策边界越趋向于直线；激活函数为tanh时，决策边界越趋向于曲线。
    X, y = make_moons(n_samples = 100, noise = 0.25, random_state = config.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y,
                                                        random_state = config.seed)

    # 默认隐层 = 1，默认隐单元 = 100，默认激活函数 = 'relu'
    for hidden_layer, active_func in [([100], 'relu'),
                                      ([100], 'tanh'),
                                      ([10], 'relu'),
                                      ([10], 'tanh'),
                                      ([10, 10], 'relu'),
                                      ([10, 10], 'tanh'),
                                      ([100, 100], 'relu'),
                                      ([100, 100], 'tanh'),
                                      ([10, 10, 10], 'relu'),
                                      ([10, 10, 10], 'tanh'),
                                      ([100, 50, 10], 'relu'),
                                      ([100, 50, 10], 'tanh'),
                                      ]:
        mlp = MLPClassifier(solver = 'lbfgs', random_state = config.seed,
                            activation = active_func, hidden_layer_sizes = hidden_layer)
        mlp.fit(X_train, y_train)

        print('-' * 20)
        title = "默认隐层={}，默认隐单元={}，默认激活函数={}".format(len(hidden_layer), hidden_layer, active_func)
        print(title)
        print('Accuracy on training set: {:.2f}'.format(mlp.score(X_train, y_train)))
        print('Accuracy on test set: {:.2f}'.format(mlp.score(X_test, y_test)))

        plt.figure()
        plt.title("图2-48：神经网络在two_moons数据集上学到的决策边界\n" + title)
        mglearn.plots.plot_2d_separator(mlp, X_train, fill = True, alpha = .3)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.legend()
        pass


def L2_regular_parameter():
    X, y = make_moons(n_samples = 100, noise = 0.25, random_state = config.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y,
                                                        random_state = config.seed)

    # 生成对比图，调节L2惩罚的参数是alpha，默认值很小（弱正则化）
    # 不同隐单元个数，值越大，模型越复杂
    # 不同alpha参数，值越大，正则化越强，模型越简单。
    fig, axes = plt.subplots(2, 4, figsize = (20, 10))
    for axx, n_hidden_nodes in zip(axes, [10, 100]):
        for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
            mlp = MLPClassifier(solver = 'lbfgs', random_state = config.seed, alpha = alpha,
                                hidden_layer_sizes = [n_hidden_nodes, n_hidden_nodes])
            mlp.fit(X_train, y_train)

            print('-' * 20)
            title = "n_hidden=[{},{}] / alpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha)
            print(title)
            print('Accuracy on training set: {:.2f}'.format(mlp.score(X_train, y_train)))
            print('Accuracy on test set: {:.2f}'.format(mlp.score(X_test, y_test)))

            mglearn.plots.plot_2d_separator(mlp, X_train, fill = True, alpha = .3, ax = ax)
            mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax = ax)
            ax.set_title(title)
            pass
        pass
    plt.suptitle("图2-52：不同隐单元个数与不同 alpha 参数的决策函数\n"
                 "默认激活函数='relu'")


# random_state 会影响神经网络对权值的初始化，从而收敛到不同的极点
def different_random_state():
    X, y = make_moons(n_samples = 100, noise = 0.25, random_state = config.seed)
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify = y, random_state = config.seed)

    # 相同参数与不同的随机初始化下的决策函数
    # 随机初始化无法控制对决策边界的影响，与数据和模型可能都有关系
    fig, axes = plt.subplots(2, 4, figsize = (15, 8))
    for i, ax in enumerate(axes.ravel()):
        mlp = MLPClassifier(solver = 'lbfgs', random_state = i ** 2, max_iter = 10000,
                            hidden_layer_sizes = [100, 100])
        mlp.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(mlp, X_train, fill = True, alpha = .3, ax = ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax = ax)
        ax.set_title('random_state = {}'.format(i ** 2))
        pass
    plt.suptitle("图2-53：相同 alpha 参数和不同随机初始化的决策函数"
                 "激活函数='relu'，隐单元=[100,100]，alpha = 0.3")


def neural_network_cancer_data():
    cancer = load_breast_cancer()

    print('=' * 20)
    print("输出Caner数据集中每个特征的最大值：")
    print('Cancer data per-feature maximal \n{}'.format(cancer.data.max(axis = 0)))

    X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, random_state = config.seed)
    mlp = MLPClassifier(random_state = config.seed)
    mlp.fit(X_train, y_train)

    print('=' * 20)
    print("原始数据经过MLP训练的结果：")
    print('Accuracy on training set: {:.2f}'.format(mlp.score(X_train, y_train)))
    print('Accuracy on test set: {:.2f}'.format(mlp.score(X_test, y_test)))

    # 数据归一化处理
    mean_on_train = X_train.mean(axis = 0)
    std_on_train = X_train.std(axis = 0)
    X_train_scaled = (X_train - mean_on_train) / std_on_train
    X_test_scaled = (X_test - mean_on_train) / std_on_train

    mlp = MLPClassifier(random_state = config.seed)
    mlp.fit(X_train_scaled, y_train)

    print('-' * 20)
    print("归一化数据经过MLP训练的结果：")
    print('Accuracy on training set: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
    print('Accuracy on test set: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))

    # 前面迭代次数不够，没有收敛，下面增加迭代次数
    mlp = MLPClassifier(random_state = config.seed, max_iter = 1000, )
    mlp.fit(X_train_scaled, y_train)

    print('-' * 20)
    print("归一化数据经过更高迭代次数的MLP训练的结果：")
    print('Accuracy on training set: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
    print('Accuracy on test set: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))
    print("数据过拟合了，尝试加入正则化控制")

    # 增加正则化影响，扩展模型泛化能力
    mlp = MLPClassifier(random_state = config.seed, max_iter = 10000, alpha = 1, )
    mlp.fit(X_train_scaled, y_train)

    print('-' * 20)
    print("归一化数据经过更高迭代次数和正则化的MLP训练的结果：")
    print('Accuracy on training set: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
    print('Accuracy on test set: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))

    # 分析网络权重，权重热图的可参考性价值不大
    plt.figure(figsize = (18, 5))
    plt.title("图2-54：神经网络在乳腺癌数据集上学到的第一个隐层权重的热图")
    plt.imshow(mlp.coefs_[0], interpolation = 'none', cmap = 'viridis')
    plt.yticks(range(30), cancer.feature_names)
    plt.xlabel('Columns in weight matrix')
    plt.ylabel('Input feature')
    plt.colorbar()
    pass


if __name__ == "__main__":
    # 图2-44：Logistic回归的可视化
    # 输入特征和预测结果显示为结点
    # 系数是结点之间的连线
    # logistic_graph()

    # 图2-45：单隐层的多层感知机可视化
    # single_hidden_layer_graph()

    # 图2-47：双隐层的多层感知机可视化
    # two_hidden_layer_graph()

    # 图2-46：两种激活函数的对比图
    # 双曲正切激活函数 和 校正线性激活函数
    # two_activate_function()

    # 神经网络的参数调整
    # tune_neural_network_parameter()

    # 图2-52：不同隐单元个数与不同 alpha 参数的决策函数
    # L2_regular_parameter()

    # 图2-53：相同 alpha 参数和不同随机初始化的决策函数
    # different_random_state()

    # 图2-54：神经网络在乳腺癌数据集上学到的第一个隐层权重的热图
    neural_network_cancer_data()

    beep_end()
    show_figures()
