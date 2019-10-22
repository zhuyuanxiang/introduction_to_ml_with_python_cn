# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C020400_classifier.py
@Version    :   v0.1
@Time       :   2019-10-05 12:25
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec0204，P91
@Desc       :   监督学习算法。分类器的不确定度估计
"""
# Chap2 监督学习
import numpy as np
import matplotlib.pyplot as plt
import mglearn
import sklearn

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf)


# 2.4. 分类器的不确定度估计（预测的置信程度）
# 即预测结果的可信度度量。
# 二分类问题可以使用决策函数和预测概率来度量预测结果的可信度。
# ToDo:决策函数和预测概率的图形不明白。
# 不同模型的决策边界的对比，以及不同模型的不确定度估计的形状。
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
def toy_data():
    # circle数据集是一个大圆，一个小圆组成的数据集
    from sklearn.datasets import make_circles
    from sklearn.model_selection import train_test_split

    X, y = make_circles()
    y_named = np.array(['blue', 'red'])[y]
    X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state = 0)

    plt.figure()
    plt.title("默认circle数据集")
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.legend()

    X, y = make_circles(noise = 0.25, factor = 0.5, random_state = 1)
    y_named = np.array(['blue', 'red'])[y]
    X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state = 0)

    plt.figure()
    plt.title("有噪声的circle数据集")
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.legend()


# 2.4.1 决策函数
def decision_function():
    # circle数据集是一个大圆，一个小圆组成的数据集
    # 准备有噪声的circle数据集
    from sklearn.datasets import make_circles
    from sklearn.model_selection import train_test_split
    X, y = make_circles(noise = 0.25, factor = 0.5, random_state = 1)
    y_named = np.array(['blue', 'red'])[y]
    X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state = 0)

    # 构建梯度提升模型
    from sklearn.ensemble import GradientBoostingClassifier
    gbdt = GradientBoostingClassifier(random_state = 0)
    gbdt.fit(X_train, y_train_named)

    # 2.4.1. 决策函数
    print('测试集的形状：{}'.format(X_test.shape))
    decision_function_values = gbdt.decision_function(X_test)
    # 二分类问题的决策函数是一维数据是历史原因造成的。
    # ToDo：什么历史原因？
    print('决策函数的形状：{}'.format(decision_function_values.shape))
    print('决策函数计算测试集的输出值：\n{}'.format(decision_function_values))
    print('决策函数计算测试集的输出值经过阈值判断的结果：\n{}'.format(decision_function_values > 0))
    print('模型预测测试集的输出结果：\n{}'.format(gbdt.predict(X_test)))

    greater_zero = (decision_function_values > 0).astype(int)
    pred = gbdt.classes_[greater_zero]

    print('决策函数输出结果与模型计算测试集的输出结果是否相等: {}'
          .format(np.all(pred == gbdt.predict(X_test))))

    decision_function_values = decision_function_values
    print('-' * 20)
    print("决策函数的输出很难解释。")
    print('Decision function minimum: {:.2f} maximum: {:.2f}'
          .format(np.min(decision_function_values), np.max(decision_function_values)))

    fig, axes = plt.subplots(1, 2, figsize = (13, 5))

    mglearn.tools.plot_2d_separator(gbdt, X, ax = axes[0], alpha = .4, fill = True, cm = mglearn.cm2)

    scores_image = mglearn.tools.plot_2d_scores(gbdt, X, ax = axes[1], alpha = .4, cm = mglearn.ReBl)

    from mglearn import discrete_scatter
    for ax in axes:
        discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers = ['^'], ax = ax)
        discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers = ['o'], ax = ax)
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')

    cbar = plt.colorbar(scores_image, ax = axes.tolist())
    axes[0].legend(['Test class 0', 'Test class 1',
                    'Train Class 0', 'Train Class 1'], ncol = 4, loc = (.1, 1.1))
    plt.title("图2-55 梯度提升模型在一个二维圆数据集上的决策边界（左）和决策函数（右）")


# 2.4.2. 预测概率
# 分类器对于大部分点都给出了相对较高的置信度。
# 不确定度的大小反映了数据通过模型和参数的计算的输出结果的不确定度。
# 过拟合更强的模型会做出置信度更高的预测，即使可能是错的。
# 过拟合更弱的模型，即复杂度更低的模型，对预测的结果的不确定度更大。
# 如果模型给出的不确定度符合实际的情况，那么这个模型就是校正（calibrated）模型。
def predict_probability():
    # 准备有噪声的circle数据集
    from sklearn.datasets import make_circles
    X, y = make_circles(noise = 0.25, factor = 0.5, random_state = 1)
    y_named = np.array(['blue', 'red'])[y]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(
            X, y_named, y, random_state = 0)

    # 构建梯度提升模型
    from sklearn.ensemble import GradientBoostingClassifier
    gbdt = GradientBoostingClassifier(random_state = 0)
    gbdt.fit(X_train, y_train_named)

    predict_proba = gbdt.predict_proba(X_test)
    # predict_result = np.array([x < y for x, y in predict_proba])
    predict_result = predict_proba.argmax(axis = 1)
    print('=' * 20)
    print('测试集的形状：{}'.format(X_test.shape))
    print('预测概率的形状：{}'.format(predict_proba.shape))

    print('-' * 20)
    print('预测概率计算测试集的输出结果:\n{}'.format(predict_proba))
    print('-' * 20)
    print('预测概率计算测试集的输出结果经过阈值判断的结果:\n{}'.format(predict_result))
    print('-' * 20)
    print('模型预测测试集的输出结果:\n{}'.format(gbdt.predict(X_test)))
    # list(map('{:.2f}%'.format, gbdt.predict_proba(X_test).ravel()))
    # narray向其他类型的转换

    greater_zero = predict_result.astype(int)
    pred = gbdt.classes_[greater_zero]
    print('-' * 20)
    print('预测概率输出结果与模型计算测试集的输出结果是否相等: {}'.format(np.all(pred == gbdt.predict(X_test))))

    # 预测概率图中的边界更加清晰。
    fig, axes = plt.subplots(1, 2, figsize = (13, 5))

    mglearn.tools.plot_2d_separator(gbdt, X, ax = axes[0], alpha = .4, fill = True, cm = mglearn.cm2)
    scores_image = mglearn.tools.plot_2d_scores(
            gbdt, X, ax = axes[1], alpha = .5, cm = mglearn.ReBl, function = 'predict_proba')

    from mglearn import discrete_scatter
    for ax in axes:
        discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers = ['^'], ax = ax)
        discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers = ['o'], ax = ax)
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')

    cbar = plt.colorbar(scores_image, ax = axes.tolist())
    axes[0].legend(['Test class 0', 'Test class 1',
                    'Train Class 0', 'Train Class 1'], ncol = 4, loc = (.1, 1.1))
    plt.title("图2-56 梯度提升模型在一个二维圆数据集上的决策边界（左）和预测概率（右）")


# 2.4.3. 多分类问题的不确定度
# 决策函数 和 预测概率 也适用于多分类问题。
def multi_classes():
    # 鸢尾花（iris）数据集是一个三分类数据集。
    from sklearn.datasets import load_iris
    iris = load_iris()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 42)

    from sklearn.ensemble import GradientBoostingClassifier
    gbdt = GradientBoostingClassifier(learning_rate = 0.01, random_state = 0)
    gbdt.fit(X_train, y_train)
    print('=' * 20)
    print("使用 GBDT 对 iris 数据集进行学习")

    # 每一列对应每个类别的“确定度分类”
    #   - 分数越高，则可能性越大
    decision_func_values = gbdt.decision_function(X_test)
    print('-' * 20)
    print('决策函数输出值的形状：{}'.format(decision_func_values.shape))
    print('决策函数的前六个输出值：\n{}'.format(decision_func_values[:6]))

    # argmax_decision_func = np.argmax(decision_func_values, axis = 1)
    argmax_decision_func = decision_func_values.argmax(axis = 1)
    print('-' * 20)
    print('决策函数的输出值中的最大项：\n{}'.format(argmax_decision_func))

    predict_prob = gbdt.predict_proba(X_test)
    print('-' * 20)
    print('预测概率输出值的形状：{}'.format(predict_prob.shape))
    print('预测概率的前6个输出值：\n{}'.format(predict_prob[:6]))

    # argmax_predict_prob = np.argmax(predict_prob, axis = 1)
    argmax_predict_prob = predict_prob.argmax(axis = 1)
    print('-' * 20)
    print('预测概率的输出值中的最大项：\n{}'.format(argmax_predict_prob))

    predict_result = gbdt.predict(X_test)
    print('-' * 20)
    print('测试数据集的预测结果：\n{}'.format(predict_result))
    print('-' * 20)
    print("决策函数的输出值中的最大项与测试数据集的预测结果是否相等？", np.all(argmax_decision_func == predict_result))
    print("预测概率的输出值中的最大项与测试数据集的预测结果是否相等？", np.all(argmax_predict_prob == predict_result))

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver = 'lbfgs', multi_class = 'auto', max_iter = 10000)
    named_target = iris.target_names[y_train]
    log_reg.fit(X_train, named_target)
    print('=' * 20)
    print("使用 LogisticRegression 对 iris 数据集进行学习")
    print('-' * 20)
    print('训练数据集中的类别：{}'.format(log_reg.classes_))

    decision_func_values = log_reg.decision_function(X_test)
    print('-' * 20)
    print('决策函数输出值的形状：{}'.format(decision_func_values.shape))
    print('决策函数的前六个输出值：')
    print(decision_func_values[:6])

    # argmax_dec_func = np.argmax(decision_func_values, axis = 1)
    argmax_dec_func = decision_func_values.argmax(axis = 1)
    print('-' * 20)
    print('决策函数的输出值中的前十个最大项：')
    print(argmax_dec_func[:10])
    print('利用分类器的classes_属性转换决策函数的输出值中的前十个最大项：')
    print(log_reg.classes_[argmax_dec_func][:10])

    predict_prob = log_reg.predict_proba(X_test)
    print('-' * 20)
    print('预测概率输出值的形状：{}'.format(predict_prob.shape))
    # argmax_predict_prob = np.argmax(predict_prob, axis = 1)
    argmax_predict_prob = predict_prob.argmax(axis = 1)
    print('-' * 20)
    print('预测概率的输出值中的前十个最大项：')
    print(argmax_predict_prob[:10])
    print('利用分类器的classes_属性转换预测概率的输出值中的前十个最大项：')
    print(log_reg.classes_[argmax_predict_prob][:10])

    predict_result = log_reg.predict(X_test)
    print('-' * 20)
    print('测试数据集的前十个预测结果：')
    print(predict_result[:10])

    print('-' * 20)
    print("决策函数的输出值中的最大项与测试数据集的预测结果是否相等？", np.all(log_reg.classes_[argmax_decision_func] == predict_result))
    print("预测概率的输出值中的最大项与测试数据集的预测结果是否相等？", np.all(log_reg.classes_[argmax_predict_prob] == predict_result))

    pass


if __name__ == "__main__":
    # circle数据集是一个大圆，一个小圆组成的数据集
    # toy_data()

    # 图2-55 梯度提升模型在一个二维圆数据集上的决策边界（左）和决策函数（右）
    # decision_function()

    # 图2-56 梯度提升模型在一个二维圆数据集上的决策边界（左）和预测概率（右）
    # predict_probability()

    # 2.4.3. 多分类问题的不确定度
    multi_classes()

    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
