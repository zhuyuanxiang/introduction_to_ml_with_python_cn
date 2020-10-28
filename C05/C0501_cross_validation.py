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
from datasets.load_data import load_train_test_blobs
from tools import *


def base_model_score():
    X_train, X_test, y_train, y_test = load_train_test_blobs()
    show_title("将数据拆分为训练集和测试集，基于 Logistic Regression 模型进行评分")

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver='lbfgs', multi_class='auto').fit(X_train, y_train)
    print("Logistic Regression 训练集的评分：{:.2f}".format(log_reg.score(X_train, y_train)))
    print("Logistic Regression 测试集的评分：{:.2f}".format(log_reg.score(X_test, y_test)))
    pass


# 5.1 交叉验证（Cross-Validation）
def plot_5_fold_data_split():
    mglearn.plots.plot_cross_validation()
    plt.suptitle("图5-1：5折交叉验证中的数据划分")


# 5.1.1 Scikit-Learn中的交叉验证
def k_fold_cross_validation():
    from sklearn.datasets import load_iris
    iris = load_iris()

    show_title("使用cross_val_score()对Logistic Regression模型进行评分")

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver='lbfgs', max_iter=10000, multi_class='auto')

    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate
    # 默认情况下，cross_val_score()执行3折交叉验证（旧的版本）
    # [0.98 0.96 0.98]-->0.9733333333333333
    # 默认情况下，cross_val_score()执行5折交叉验证（新的版本）
    # [0.967 1.    0.933 0.967 1.   ]-->0.9733333333333334
    # 折与折之间精度变化较大，意味着模型对于某个折出现过拟合，说明数据分布不够随机，数据量需要增大
    # 折与折之间精度变化较大，意味着模型对于某个折出现过拟合，说明数据分布不够随机，数据量需要增大
    for cv in [3, 4, 5]:
        scores = cross_val_score(log_reg, iris.data, iris.target, cv=cv)
        show_subtitle(f"LogisticRegression 使用 iris 数据集经过 {cv} 折交叉验证")
        print(f"每一折切分数据得到的模型精度：{scores}")
        print(f"交叉验证的平均值：{scores.mean()}")
        res = cross_validate(log_reg, iris.data, iris.target, cv=cv, return_train_score=True)
        print("res =", res)
        res_df = pd.DataFrame(res)
        print(res_df)
        print("Mean times and scores:\n", res_df.mean())


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
    print("原始数据的类别是排序的")
    print("iris.data = ", iris.data)
    print("iris.target = ", iris.target)

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)

    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score

    # KFold()是不分层的分离器
    # 默认情况下，cross_val_score()执行不分层3折交叉验证（旧的版本）
    # [0. 0. 0.]，排序数据不分层交叉验证的结果非常糟糕
    # 默认情况下，cross_val_score()执行不分层5折交叉验证（新的版本）
    # [1.    1.    0.867 0.933 0.833] --> 0.927
    # 折与折之间精度变化较大，意味着模型对于某个折出现过拟合，说明数据分布不够随机，数据量需要增大
    show_title("使用 KFold() 基于 iris 排序数据的数据集，不分层对Logistic Regression模型进行评分")
    for n_splits in [3, 4, 5]:
        k_fold = KFold(n_splits=n_splits)
        scores = cross_val_score(log_reg, iris.data, iris.target, cv=k_fold)
        show_subtitle(f" {n_splits} 折交叉验证")
        print(f"模型的预测结果：{scores}")
        print(f"模型的平均值：{scores.mean()}")

    # 打乱数据不分层 3 折交叉验证：[0.98 0.96 0.96]-->0.967
    # 打乱数据不分层 5 折交叉验证：[1.    0.833 1.    1.    0.933]-->0.953
    # 5 折没有 3 折好，5 折的精度变化大，说明数据量不够，5 折分隔数据造成部分集合过分隔过拟合
    show_title("使用 KFold() 基于 iris 打乱数据的数据集，不分层对Logistic Regression模型进行评分")
    for n_splits in [3, 4, 5]:
        k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(log_reg, iris.data, iris.target, cv=k_fold)
        show_subtitle(f" {n_splits} 折交叉验证")
        print(f"模型的预测结果：{scores}")
        print(f"模型的平均值：{scores.mean()}")

    # cv=数字时，默认使用 分层K折交叉验证 （StratifiedKFold）
    show_title("使用 cross_val_score() 基于 iris 排序数据的数据集，纯数字（默认分层）对Logistic Regression模型进行评分")
    for n_splits in [3, 4, 5]:
        scores = cross_val_score(log_reg, iris.data, iris.target, cv=n_splits)
        show_subtitle(f" {n_splits} 折交叉验证")
        print(f"模型的预测结果：{scores}")
        print(f"模型的平均值：{scores.mean()}")

    show_title("使用 cross_val_score() 基于 iris 排序数据的数据集，分层对Logistic Regression模型进行评分")
    for n_splits in [3, 4, 5]:
        k_fold = StratifiedKFold(n_splits=n_splits)
        scores = cross_val_score(log_reg, iris.data, iris.target, cv=k_fold)
        show_subtitle(f" {n_splits} 折交叉验证")
        print(f"模型的预测结果：{scores}")
        print(f"模型的平均值：{scores.mean()}")


# 2. 留一法（leave-one-out）：保留一个数据，效果很好，非常耗时，适合小型数据集。
# 留P法（LeavePOut)：保留P个数据
def cross_validation_splitter_leave_one_out():
    from sklearn.datasets import load_iris
    iris = load_iris()

    show_title("使用留一法对Logistic Regression模型进行评分")

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)

    from sklearn.model_selection import LeaveOneOut
    leave_one_out = LeaveOneOut()

    from sklearn.model_selection import cross_val_score
    # 默认情况下，cross_val_score()执行分层5折交叉验证（新的版本）
    scores = cross_val_score(log_reg, iris.data, iris.target, cv=leave_one_out)
    print("LogisticRegression 使用 iris 数据集经过留一交叉验证的结果：{}".format(scores))
    # 有1也有0，说明某些验证失败了。->0.9666666666666667
    # ToDo:为什么会失效呢？是不是验证集中的数据风格在训练集中不存在？
    print("LogisticRegression 使用 iris 数据集经过留一交叉验证的平均值：{}".format(scores.mean()))
    pass


# 3. 打乱划分交叉验证
# 分层打乱划分交叉验证（StratifiedShuffleSplit）
def plot_shuffle_split():
    mglearn.plots.plot_shuffle_split()
    plt.title("图5-3：对10个点进行打乱划分，参数 train_size=5, test_size=2, n_iter=4")


def cross_validation_splitter_shuffle_split():
    from sklearn.datasets import load_iris
    iris = load_iris()

    show_title("使用打乱划分交叉验证对Logistic Regression模型进行评分")

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)

    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import cross_val_score

    for train_size in [.3, .5, .7]:
        shuffle_split = ShuffleSplit(train_size=train_size, test_size=round(1 - train_size, 1), n_splits=10)
        # 默认情况下，cross_val_score()执行分层3折交叉验证（旧的版本）
        scores = cross_val_score(log_reg, iris.data, iris.target, cv=shuffle_split)
        print(f"LogisticRegression 使用 {train_size * 100}% iris 数据集经过打乱划分交叉验证的结果：{scores}")
        # [0.978 0.933 0.956 0.978 0.956 0.956 0.956 0.956 0.956 0.956]->0.9577777777777777
        print(f"LogisticRegression 使用 {train_size * 100}% iris 数据集经过打乱划分交叉验证的平均值：{scores.mean()}")
        pass


# 4.分组交叉验证（ShuffleSplit）
# 以groups数据作为分组的依据。
# 分组打乱划分交叉验证（GroupShuffleSplit）
# 分层打乱划分交叉验证（StratifiedShuffleSplit）（参考3）
def cross_validation_splitter_group_k_fold():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=12, random_state=0)

    show_title("使用分组打乱划分交叉验证对 Logistic Regression 模型进行评分")

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)

    # 定义：0，1，2三个样本属于0组；3，4，5，6样本属于1组；7，8样本属于2组；9，10，11样本属于3组
    groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]

    from sklearn.model_selection import GroupKFold
    from sklearn.model_selection import cross_val_score

    # 默认情况下，cross_val_score()执行分层3折交叉验证
    for n_splits in [2, 3, 4]:
        group_k_fold = GroupKFold(n_splits=n_splits)
        scores = cross_val_score(log_reg, X, y, groups, cv=group_k_fold)
        # 2: [0.667 0.5   ] -> 0.5833333333333333
        # 3: [0.75  0.6   0.667]->0.6722222222222222
        # 4：[0.75  0.667 0.667 1.   ]->0.7708333333333333
        print("LogisticRegression 使用 blobs 数据集经过分组(3组）交叉验证的结果：{}".format(scores))
        print("LogisticRegression 使用 blobs 数据集经过分组(3组）交叉验证的平均值：{}".format(scores.mean()))
        pass


def group_k_fold_data_distribution():
    mglearn.plots.plot_group_kfold()
    plt.title("图5-4：用GroupKFold进行依赖于标签的划分")


def main():
    # 手工拆分数据为训练集和测试集对Logistic Regression模型进行评分
    # base_model_score()
    # 使用cross_val_score()对Logistic Regression模型进行评分
    # k_fold_cross_validation()
    # Scikit-Learn中的交叉验证的各种策略
    # compare_cross_validation_data_distribution()
    # 1. 不分层K折交叉验证分离器（KFold）
    # cross_validation_splitter_k_fold()
    # 2. 留一法（leave-one-out）：保留一个数据，效果很好，非常耗时，适合小型数据集。
    # cross_validation_splitter_leave_one_out()
    # 3. 打乱划分交叉验证
    # 分层打乱划分交叉验证（StratifiedShuffleSplit）
    # plot_shuffle_split()
    # cross_validation_splitter_shuffle_split()
    # 4.分组交叉验证（ShuffleSplit）
    # 分组打乱划分交叉验证（GroupShuffleSplit）
    # cross_validation_splitter_group_k_fold()
    # group_k_fold_data_distribution()
    pass


if __name__ == "__main__":
    main()
    beep_end()
    show_figures()
