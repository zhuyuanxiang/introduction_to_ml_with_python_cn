# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C0107.py
@Version    :   v0.1
@Time       :   2019-09-17 18:22
@License    :   (C)Copyright 2019-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Ch0107，P11
@Desc       :   引言。第一个应用：鸢尾花分类
"""
import config
import mglearn
import numpy as np
import pandas as pd
import sklearn


# 1.7. 鸢尾花分类
# 分步骤执行，仔细观察系统的功能和数据的内涵，更加深入的理解对于后面的学习会有帮助
def train_iris_segment():  # 在PyCharm中使用Alt+Shift+E一条条语句在Python Console中执行，观察结果
    # 1.7.1. 导入鸢尾花的数据
    iris_dataset = sklearn.datasets.load_iris()
    print('Keys of iris_database: \n{}'.format(iris_dataset.keys()))
    print(iris_dataset['DESCR'][:193] + '\n...')
    print('Target names: {}'.format(iris_dataset['target_names']))
    print('Feature names:\n{}'.format(iris_dataset['feature_names']))
    print('Type of data: {}'.format(type(iris_dataset['data'])))
    print('Shape of data: {}'.format(iris_dataset['data'].shape))
    print('First five rows of data:\n{}'.format(iris_dataset['data'][:5]))
    print('Type of target: {}'.format(type(iris_dataset['target'])))
    print('Shape of target: {}'.format(iris_dataset['target'].shape))
    print('Target:\n{}'.format(iris_dataset['target']))

    # 1.7.2. 准备数据
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            iris_dataset['data'], iris_dataset['target'], random_state = config.seed)
    print('X_train shape: {}'.format(X_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    print('y_test shape: {}'.format(y_test.shape))

    # 1.7.3. 观察数据(ToDo: 对数据散点图的认识还需要加强）
    iris_dataframe = pd.DataFrame(X_train, columns = iris_dataset.feature_names)
    grr = pd.plotting.scatter_matrix(
            iris_dataframe, c = y_train, figsize = (15, 15),
            marker = 'o', hist_kwds = {'bins': 20}, s = 60, alpha = 0.8, cmap = mglearn.cm3)

    # 1.7.4. K近邻算法
    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 1)
    model.fit(X_train, y_train)  # 模型训练

    # 1.7.5. 做出预测
    X_new = np.array([[5, 2.9, 1, 0.2]])
    print('X_new.shape: {}'.format(X_new.shape))
    prediction = model.predict(X_new)
    print('Prediction: {}'.format(prediction))
    print('Predicted target name: {}'.format(iris_dataset['target_names'][prediction]))

    # 1.7.6. 模型评估：精确度（accuracy）衡量模型的优劣
    y_pred = model.predict(X_test)
    # 手工计算预测的结果和精确度
    print('Test set predictions:\n{}'.format(y_pred))
    print('Test set score: {:.2f}'.format(np.mean(y_pred == y_test)))
    print('Test set score: {:.2f}'.format(np.count_nonzero(y_pred == y_test) / len(y_test)))
    # 模型评估预测的精确度
    print('Test set score: {:.2f}'.format(model.score(X_test, y_test)))

    pass


# 鸢尾花分类的完整运行
def train_iris_completion():
    iris_dataset = sklearn.datasets.load_iris()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            iris_dataset['data'], iris_dataset['target'], random_state = config.seed)
    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 1)
    model.fit(X_train, y_train)  # 模型训练
    print('Test set score: {:.2f}'.format(model.score(X_test, y_test)))
    pass


if __name__ == "__main__":
    # 鸢尾花分类的分步骤运行
    train_iris_segment()

    # 鸢尾花分类的完整运行
    train_iris_completion()

    import tools
    tools.beep_end()
    tools.show_figures()
