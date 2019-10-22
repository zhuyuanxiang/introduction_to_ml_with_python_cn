# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C0405_auto_select_feature.py
@Version    :   v0.1
@Time       :   2019-10-10 09:55
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec0405，P181
@Desc       :   数据表示与特征工程。自动选择特征。
"""
import matplotlib.pyplot as plt
import numpy as np

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 5, suppress = True, threshold = np.inf, linewidth = 200)


# 4.5. 自动化特征选择（基于特征的影响力）
# 4.5.1. 单变量统计（计算每个特征和目标值之间的关系是否存在统计显著性，然后选择最高置信度的特征）
# 对于分类问题，也称为方差分析；关键性质是单变量的，即只单独考虑每个特征
# 计算速度快，不需要构建模型；完全独立于特征选择之后的应用模型
def univariate_statistics():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    rng = np.random.RandomState(42)
    noise = rng.normal(size = (len(cancer.data), 50))
    # 在数据中添加噪声特征（前30个特征来自数据集，后50个特征来自噪声），形成噪声数据
    X_w_noise = np.hstack([cancer.data, noise])

    # 从噪声数据中选择一半的数据作为训练集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state = 0, test_size = .5)
    # 从噪声数据中选择一部分特征作为训练数据的特征
    # 特征越少，精确度越高（可能是噪声干扰越少）
    fig, axes = plt.subplots(4, 1, figsize = (15, 9))
    plt.suptitle("图4-9：SelectPercentile选择的特征")
    # 在Scikit - Learn中
    # score_func=f_classif（默认值），用于分类问题；
    # score_func=f_regression，用于回归问题
    # SelectPercentile(score_func = ?, percentile =)：选择固定百分比的特征
    # SelectKBest(score_func = ?, k =)：选择固定数目的特征
    from sklearn.feature_selection import SelectPercentile
    for ax, percentile in zip(axes.ravel(), [10, 25, 50, 75]):
        select = SelectPercentile(percentile = percentile)
        select.fit(X_train, y_train)
        X_train_selected = select.transform(X_train)
        print('-' * 20)
        print("特征比例（percentile）={}".format(percentile))
        print('X_train.shape: {}'.format(X_train.shape))
        print('X_train_selected.shape: {}'.format(X_train_selected.shape))

        # 检测哪些特征被选中
        mask = select.get_support()
        # print(mask)

        # 将选中的特征可视化，噪声数据中的某些原始特征在特征选择中被放弃
        ax.matshow(mask.reshape(1, -1), cmap = 'gray_r')
        ax.set_xlabel("Sample Index\n特征数目= {}".format(X_train_selected.shape[1]))

        # 删除特征反而能够提高性能，哪怕丢失了某些原始特征以后。
        from sklearn.linear_model import LogisticRegression

        X_test_selected = select.transform(X_test)
        lr = LogisticRegression(solver = 'lbfgs', max_iter = 10000)
        lr.fit(X_train, y_train)
        print('Score with all features: {:.3f}'.format(lr.score(X_test, y_test)))
        lr.fit(X_train_selected, y_train)
        print('Score with only selected features: {:.3f}'.format(lr.score(X_test_selected, y_test)))
        pass


# 4.5.2. 基于模型的特征选择（使用监督机器学习模型来判断每个特征的重要性，仅保留最重要的特征）
def select_features_based_on_model():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    rng = np.random.RandomState(42)
    noise = rng.normal(size = (len(cancer.data), 50))
    # 在数据中添加噪声特征（前30个特征来自数据集，后50个特征来自噪声），形成噪声数据
    X_w_noise = np.hstack([cancer.data, noise])

    # 从噪声数据中选择一半的数据作为训练集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state = 0, test_size = .5)

    # 从噪声数据中选择一部分特征作为训练数据的特征
    # 特征越少，精确度越高（可能是噪声干扰越少）
    fig, axes = plt.subplots(4, 1, figsize = (15, 9))
    plt.suptitle("图4-10：使用RandomForestClassifier的SelectFromModel选择的特征")
    # 使用SelectFromModel变换器从模型中选出需要的特征
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    # 使用不同的模型来学习，结果肯定不同，
    # 使用随机森林学习时会更好的使用数据的全局关系，
    # 使用Logistic Regression模型学习时，就只能使用每个特征自己的内在关系，
    # 所以充分学习后，就会因为有用的特征的评分不高而被放弃，导致最终评分模型学习结果较差
    # ToDo：学习特征的模型和学习数据的模型是一样时，这样的设计目的我认为主要是为了降维。
    for ax, feature_number in zip(axes.ravel(), [8, 20, 40, 60]):
        # select = SelectFromModel(RandomForestClassifier(n_estimators = 100, random_state = 42), threshold = 'median')
        # select = SelectFromModel(RandomForestClassifier(n_estimators = 100, random_state = 42),
        #                          max_features = feature_number, threshold = -np.inf)

        # 注意对比下面两个训练结果，Logistic Regression充分学习后选择的特征使得学习模型效果变差
        # select = SelectFromModel(LogisticRegression(solver = 'lbfgs',random_state = 42),
        #                          max_features = feature_number, threshold = -np.inf)
        select = SelectFromModel(LogisticRegression(solver = 'lbfgs', max_iter = 10000, random_state = 42),
                                 max_features = feature_number, threshold = -np.inf)
        select.fit(X_train, y_train)
        X_train_l1 = select.transform(X_train)
        X_test_l1 = select.transform(X_test)
        print('-' * 20)
        print('X_train.shape: {}'.format(X_train.shape))
        print('X_train_l1.shape: {}'.format(X_train_l1.shape))

        # 检测哪些特征被选中
        mask = select.get_support()
        # print(mask)

        # 将选中的特征可视化，噪声数据中的某些原始特征在特征选择中被放弃
        ax.matshow(mask.reshape(1, -1), cmap = 'gray_r')
        ax.set_xlabel('Sample Index\n特征数目= {}'.format(feature_number))

        # 删除特征反而能够提高性能，哪怕丢失了某些原始特征以后。
        lr = LogisticRegression(solver = 'lbfgs', max_iter = 10000)
        lr.fit(X_train, y_train)
        print('Score with all features: {:.3f}'.format(lr.score(X_test, y_test)))
        lr.fit(X_train_l1, y_train)
        print('Score with only selected features: {:.3f}'.format(lr.score(X_test_l1, y_test)))
        # 模型选择的特征质量更好，选择更多特征也能够得到较好的精确度
        pass


# 4.5.3. 迭代特征选择：构建一系列模型，每个模型都使用不同数量的特征
# 递归特征消除（Recursive Feature Elimination，RFE）：从所有特征开始构建模型，根据模型舍弃最不重要的特征，直到满足某个终止条件。
# 速度较慢，效果一般。
def iterative_selection():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    rng = np.random.RandomState(42)
    noise = rng.normal(size = (len(cancer.data), 50))
    # 在数据中添加噪声特征（前30个特征来自数据集，后50个特征来自噪声），形成噪声数据
    X_w_noise = np.hstack([cancer.data, noise])

    # 从噪声数据中选择一半的数据作为训练集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state = 0, test_size = .5)

    # 从噪声数据中选择一部分特征作为训练数据的特征
    # 特征越少，精确度越高（可能是噪声干扰越少）
    fig, axes = plt.subplots(4, 1, figsize = (15, 9))
    plt.suptitle("图4-11：使用随机森林分类器模型的递归特征消除选择的特征")
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    for ax, feature_number in zip(axes.ravel(), [8, 20, 40, 60]):
        # select = RFE(RandomForestClassifier(n_estimators = 100, random_state = 42),
        #              n_features_to_select = feature_number)

        # 注意对比下面两个训练结果，Logistic Regression充分学习后选择的特征使得学习模型效果变差
        # ToDo:不充分地学习，选择40个特征后，学习模型的得分会得到显著提升（有趣），为什么？
        select = RFE(LogisticRegression(solver = 'lbfgs', random_state = 42),
                     n_features_to_select = feature_number)
        # select = RFE(LogisticRegression(solver = 'lbfgs', max_iter = 10000, random_state = 42),
        #              n_features_to_select = feature_number)
        select.fit(X_train, y_train)
        X_train_rfe = select.transform(X_train)
        X_test_rfe = select.transform(X_test)
        print('-' * 20)
        print('X_train.shape: {}'.format(X_train.shape))
        print('X_train_rfe.shape: {}'.format(X_train_rfe.shape))

        # 检测哪些特征被选中
        mask = select.get_support()
        # print(mask)

        # 将选中的特征可视化，噪声数据中的某些原始特征在特征选择中被放弃
        ax.matshow(mask.reshape(1, -1), cmap = 'gray_r')
        ax.set_xlabel('Sample Index\n特征数目= {}'.format(feature_number))

        # 删除特征反而能够提高性能，哪怕丢失了某些原始特征以后。
        lr = LogisticRegression(solver = 'lbfgs', max_iter = 10000)
        lr.fit(X_train, y_train)
        print('Score with all features: {:.3f}'.format(lr.score(X_test, y_test)))
        lr.fit(X_train_rfe, y_train)
        print('Score with only selected features: {:.3f}'.format(lr.score(X_test_rfe, y_test)))


if __name__ == "__main__":
    """对比三种自动化选择特征的方法的图形，可以发现原始特征最多的时候不一定精确度最高，
    估计某些原始特征是噪声，对精确度的提高没有贡献。"""
    # 4.5.1. 单变量统计（计算每个特征和目标值之间的关系是否存在统计显著性，然后选择最高置信度的特征）
    # univariate_statistics()

    # 4.5.2. 基于模型的特征选择（使用监督机器学习模型来判断每个特征的重要性，仅保留最重要的特征）
    # select_features_based_on_model()

    # 4.5.3. 迭代特征选择：构建一系列模型，每个模型都使用不同数量的特征
    iterative_selection()
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
