# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C0501_cross_validation.py
@Version    :   v0.1
@Time       :   2019-10-10 11:34
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec0501，P194
@Desc       :   模型评估与改进
"""
import matplotlib.pyplot as plt
import mglearn
import numpy as np

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)


def base_model_score():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(random_state = 0)

    # 将数据划分为训练集和测试集，是为了利用测试集度量模型的泛化能力。
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    number_title = "手工拆分数据为训练集和测试集对Logistic Regression模型进行评分"
    print('\n', '-' * 5, number_title, '-' * 5)

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver = 'lbfgs', multi_class = 'auto').fit(X_train, y_train)
    print("LogisticRegression测试集的评分：{:.2f}".format(log_reg.score(X_test, y_test)))
    pass


# 5.1 交叉验证（Cross-Validation）
# 5.1.1 Scikit-Learn中的交叉验证
def k_fold_cross_validation():
    from sklearn.datasets import load_iris
    iris = load_iris()

    number_title = "使用cross_val_score()对Logistic Regression模型进行评分"
    print('\n', '-' * 5, number_title, '-' * 5)

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver = 'lbfgs', max_iter = 10000, multi_class = 'auto')

    from sklearn.model_selection import cross_val_score
    # 默认情况下，cross_val_score()执行3折交叉验证（旧的版本）
    # [0.98  0.941 1.   ]-->0.974
    # 默认情况下，cross_val_score()执行5折交叉验证（新的版本）
    # [0.967 1.    0.933 0.967 1.   ]-->0.973
    # 折与折之间精度变化较大，意味着模型对于某个折出现过拟合，说明数据分布不够随机，数据量需要增大
    for cv in [3, 5]:
        scores = cross_val_score(log_reg, iris.data, iris.target, cv = cv)
        print("LogisticRegression 使用 iris 数据集经过 {} 折交叉验证的结果：{}".format(cv, scores))
        print("LogisticRegression 使用 iris 数据集经过 {} 折交叉验证的平均值：{}".format(cv, scores.mean()))
        pass
    pass


# 5.1.3 Scikit-Learn中的交叉验证的各种策略
def compare_cross_validation_data_distribution():
    mglearn.plots.plot_stratified_cross_validation()
    plt.suptitle("图5-2：当数据按照类别标签排序时，标准交叉验证与分层交叉验证的对比")


# 使用交叉验证分离器（Cross-Validation Splitter）可以对数据划分过程进行更加精细地控制

# 默认情况下使用的是分层K折交叉验证分离器(StratifiedKFold)
# Scikit-Learn还有重复分层K折交叉验证分离器（RepeatedStratifiedKFold），进一步增加数据分布的随机性
# 1. 不分层K折交叉验证分离器（KFold）
def cross_validation_splitter_k_fold():
    from sklearn.datasets import load_iris
    iris = load_iris()

    number_title = "使用KFold()不分层对Logistic Regression模型进行评分"
    print('\n', '-' * 5, number_title, '-' * 5)

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver = 'lbfgs', multi_class = 'auto', max_iter = 10000)

    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score

    # KFold()是不分层的分离器
    # 默认情况下，cross_val_score()执行不分层3折交叉验证（旧的版本）
    # [0. 0. 0.]，排序数据不分层交叉验证的结果非常糟糕
    # 默认情况下，cross_val_score()执行不分层5折交叉验证（新的版本）
    # [1.    1.    0.867 0.933 0.833] --> 0.927
    # 折与折之间精度变化较大，意味着模型对于某个折出现过拟合，说明数据分布不够随机，数据量需要增大
    for n_splits in [3, 5]:
        k_fold = KFold(n_splits = n_splits)
        scores = cross_val_score(log_reg, iris.data, iris.target, cv = k_fold)
        print("LogisticRegression 使用 iris 数据集经过不分层 {} 折交叉验证的结果：{}".format(n_splits, scores))
        print("LogisticRegression 使用 iris 数据集经过不分层 {} 折交叉验证的平均值：{}".format(n_splits, scores.mean()))
        pass
    pass

    number_title = "使用KFold()打乱数据不分层对Logistic Regression模型进行评分"
    # 打乱数据不分层 3 折交叉验证：[0.98 0.96 0.96]-->0.967
    # 打乱数据不分层 5 折交叉验证：[1.    0.833 1.    1.    0.933]-->0.953
    # 5 折没有 3 折好，5 折的精度变化大，说明数据量不够，5 折分隔数据造成部分集合过分隔过拟合
    print('\n', '-' * 5, number_title, '-' * 5)
    for n_splits in [3, 5]:
        k_fold = KFold(n_splits = n_splits, shuffle = True, random_state = 0)
        scores = cross_val_score(log_reg, iris.data, iris.target, cv = k_fold)
        print("LogisticRegression 使用 iris 数据集经过打乱数据不分层 {} 折交叉验证的结果：{}".format(n_splits, scores))
        print("LogisticRegression 使用 iris 数据集经过打乱数据不分层 {} 折交叉验证的平均值：{}".format(n_splits, scores.mean()))
        pass
    pass


# 2. 留一法（leave-one-out）：保留一个数据，效果很好，非常耗时，适合小型数据集。
# 留P法（LeavePOut)：保留P个数据
def cross_validation_splitter_leave_one_out():
    from sklearn.datasets import load_iris
    iris = load_iris()

    number_title = "使用留一法对Logistic Regression模型进行评分"
    print('\n', '-' * 5, number_title, '-' * 5)

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver = 'lbfgs', multi_class = 'auto', max_iter = 10000)

    from sklearn.model_selection import LeaveOneOut
    leave_one_out = LeaveOneOut()

    from sklearn.model_selection import cross_val_score
    # 默认情况下，cross_val_score()执行分层5折交叉验证（新的版本）
    scores = cross_val_score(log_reg, iris.data, iris.target, cv = leave_one_out)
    print("LogisticRegression 使用 iris 数据集经过留一交叉验证的结果：{}".format(scores))
    # 有1也有0，说明某些验证失败了。->0.9666666666666667
    print("LogisticRegression 使用 iris 数据集经过留一交叉验证的平均值：{}".format(scores.mean()))
    pass


# 3. 打乱划分交叉验证
# 分层打乱划分交叉验证（StratifiedShuffleSplit）
def cross_validation_splitter_shuffle_split():
    from sklearn.datasets import load_iris
    iris = load_iris()

    number_title = "使用打乱划分交叉验证对Logistic Regression模型进行评分"
    print('\n', '-' * 5, number_title, '-' * 5)

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver = 'lbfgs', multi_class = 'auto', max_iter = 10000)

    from sklearn.model_selection import ShuffleSplit
    shuffle_split = ShuffleSplit(train_size = .7, test_size = .3, n_splits = 10)

    from sklearn.model_selection import cross_val_score
    # 默认情况下，cross_val_score()执行分层3折交叉验证（旧的版本）
    scores = cross_val_score(log_reg, iris.data, iris.target, cv = shuffle_split)
    print("LogisticRegression 使用 70% iris 数据集经过打乱划分交叉验证的结果：{}".format(scores))
    # [0.978 0.933 0.956 0.978 0.956 0.956 0.956 0.956 0.956 0.956]->0.9577777777777777
    print("LogisticRegression 使用 70% iris 数据集经过打乱划分交叉验证的平均值：{}".format(scores.mean()))

    shuffle_split = ShuffleSplit(train_size = .5, test_size = .5, n_splits = 10)
    # 默认情况下，cross_val_score()执行分层3折交叉验证（旧的版本）
    scores = cross_val_score(log_reg, iris.data, iris.target, cv = shuffle_split)
    print("LogisticRegression 使用 50% iris 数据集经过打乱划分交叉验证的结果：{}".format(scores))
    # [0.96  0.92  0.973 0.893 0.973 0.987 0.96  0.933 0.96  0.973]->0.9533333333333331
    print("LogisticRegression 使用 50% iris 数据集经过打乱划分交叉验证的平均值：{}".format(scores.mean()))

    pass


# 4.分组交叉验证（ShuffleSplit）
# 以groups数据作为分组的依据。
# 分组打乱划分交叉验证（GroupShuffleSplit）
# 分层打乱划分交叉验证（StratifiedShuffleSplit）（参考3）
def cross_validation_splitter_group_k_fold():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples = 12, random_state = 0)

    number_title = "使用分组打乱划分交叉验证对 Logistic Regression 模型进行评分"
    print('\n', '-' * 5, number_title, '-' * 5)

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver = 'lbfgs', multi_class = 'auto', max_iter = 10000)

    # 定义：0，1，2三个样本属于0组；3，4，5，6样本属于1组；7，8样本属于2组；9，10，11样本属于3组
    groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]

    from sklearn.model_selection import GroupKFold
    from sklearn.model_selection import cross_val_score

    group_k_fold = GroupKFold(n_splits = 3)
    # 默认情况下，cross_val_score()执行分层3折交叉验证
    scores = cross_val_score(log_reg, X, y, groups, cv = group_k_fold)
    print("LogisticRegression 使用 blobs 数据集经过分组(3组）交叉验证的结果：{}".format(scores))
    # [0.75  0.6   0.667]->0.6722222222222222
    print("LogisticRegression 使用 blobs 数据集经过分组(3组）交叉验证的平均值：{}".format(scores.mean()))

    group_k_fold = GroupKFold(n_splits = 4)
    # 默认情况下，cross_val_score()执行分层3折交叉验证
    scores = cross_val_score(log_reg, X, y, groups, cv = group_k_fold)
    print("LogisticRegression 使用 blobs 数据集经过分组(4组）交叉验证的结果：{}".format(scores))
    # [0.75  0.667 0.667 1.   ]->0.7708333333333333
    print("LogisticRegression 使用 blobs 数据集经过分组(4组）交叉验证的平均值：{}".format(scores.mean()))

    pass


def group_k_fold_data_distribution():
    # mglearn.plots.plot_group_kfold()
    from sklearn.model_selection import GroupKFold
    groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]

    plt.figure(figsize = (10, 2))
    plt.title("GroupKFold")

    axes = plt.gca()
    axes.set_frame_on(False)

    n_folds = 12
    n_samples = 12
    n_iter = 3
    n_samples_per_fold = 1

    cv = GroupKFold(n_splits = 3)
    mask = np.zeros((n_iter, n_samples))
    for i, (train, test) in enumerate(cv.split(range(12), groups = groups)):
        mask[i, train] = 1
        mask[i, test] = 2

    for i in range(n_folds):
        # test is grey
        colors = ["grey" if x == 2 else "white" for x in mask[:, i]]
        # not selected has no hatch

        # 画带斜杠的方框
        boxes = axes.barh(y = range(1, n_iter + 1), width = [1 - 0.1] * n_iter, height = .6,
                          left = i * n_samples_per_fold, color = colors,
                          hatch = "//", edgecolor = "k", align = 'edge')
        for j in np.where(mask[:, i] == 0)[0]:
            boxes[j].set_hatch("")

    # 画空的方框
    axes.barh(y = [n_iter + 1] * n_folds, width = [1 - 0.1] * n_folds, height = .6,
              left = np.arange(n_folds) * n_samples_per_fold,
              color = "w", edgecolor = 'k', align = "edge")

    for i in range(12):
        axes.text((i + .5) * n_samples_per_fold, 4.5, "%d" %
                  groups[i], horizontalalignment = "center")

    axes.invert_yaxis()
    axes.set_xlim(0, n_samples + 1)
    axes.set_ylabel("CV iterations")
    axes.set_xlabel("Data points")
    axes.set_xticks(np.arange(n_samples) + .5)
    axes.set_xticklabels(np.arange(1, n_samples + 1))
    # 设置y坐标轴的刻度
    axes.set_yticks(np.arange(0, n_iter + 3) + .3)
    # y坐标轴刻度的标识
    axes.set_yticklabels([""] + ["Split %d" % x for x in range(1, n_iter + 1)] + ["Group"])
    plt.legend([boxes[0], boxes[1]], ["Training set", "Test set"], loc = (1, .3))
    plt.tight_layout()
    plt.suptitle("图5-4：用GroupKFold进行依赖于标签的划分")


if __name__ == "__main__":
    # 手工拆分数据为训练集和测试集对Logistic Regression模型进行评分
    # base_model_score()

    # 使用cross_val_score()对Logistic Regression模型进行评分
    # k_fold_cross_validation()

    # Scikit-Learn中的交叉验证的各种策略
    # 我的电脑显示效果不好，还是参考书上的图吧。
    # compare_cross_validation_data_distribution()

    # 1. 不分层K折交叉验证分离器（KFold）
    # cross_validation_splitter_k_fold()

    # 2. 留一法（leave-one-out）：保留一个数据，效果很好，非常耗时，适合小型数据集。
    # cross_validation_splitter_leave_one_out()

    # 3. 打乱划分交叉验证
    # 分层打乱划分交叉验证（StratifiedShuffleSplit）
    # cross_validation_splitter_shuffle_split()

    # 4.分组交叉验证（ShuffleSplit）
    # 分组打乱划分交叉验证（GroupShuffleSplit）
    # cross_validation_splitter_group_k_fold()

    group_k_fold_data_distribution()
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
