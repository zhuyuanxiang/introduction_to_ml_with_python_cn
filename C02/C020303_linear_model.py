# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C020303_linear_model.py
@Version    :   v0.1
@Time       :   2019-10-15 9:38
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec020303，P35
@Desc       :   监督学习算法。线性模型
"""

# 2.3. 监督学习算法
import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from config import seed
from datasets.load_data import load_train_test_extended_boston
from datasets.load_data import load_train_test_wave
from datasets.load_data import make_my_forge
from tools import show_title


# 2.3.3. 线性模型
# 1）用于回归的线性模型
# - 单一特征的预测结果是一条直线
# - 两个特征的预测结果是一个平面
# - 多个特征的预测结果是一个超平面
# 如果特征数量大于训练数据点的数量，任何目标y都可以（在训练集上）用线性函数完美拟合。（参考线性代数）
def linear_model_regression():
    mglearn.plots.plot_linear_regression_wave()
    plt.suptitle("图2-11：线性模型对wave数据集的预测结果")


# 2）线性回归，又叫最小二乘法（Least Squares Method，LSM），
# 目的是预测值和真实值之间的差的平方和，即均方误差最小。
# 均方误差(Mean Squared Error，MSE)：预测值与真实值之间的平方和除以样本数。
# -   优点：线性回归没有需要设置的参数，即模型没有超参数
# -   缺点：无法控制模型的复杂度
# “斜率”参数，也叫做权重，或者系数，保存在coef_属性中；
# “偏移“参数，也叫截距，保存在intercept_属性中。
# coef_和intercept_结尾处的下划线：表示从训练数据中计算得到的值，用于区分用户设置的参数
def least_squares():
    X_train, X_test, y_train, y_test = load_train_test_wave(n_samples=60)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(X_train, y_train)
    print(f'线性回归模型的系数：{lr.coef_}')
    print(f'线性回归模型的偏移：{lr.intercept_}')
    # 训练集与测试集的得分非常接近，并且模型精度较低，说明训练存在欠拟合。
    print('训练集得分: {:.2f}'.format(lr.score(X_train, y_train)))
    print('测试集得分: {:.2f}'.format(lr.score(X_test, y_test)))


# 最小二乘法应用于高维数据
# 扩展的 Boston 数据
def least_squares_high_dimension():
    X_train, X_test, y_train, y_test = load_train_test_extended_boston()

    lr = LinearRegression().fit(X_train, y_train)
    print(f'线性回归模型的系数：{lr.coef_}')
    print(f'线性回归模型的偏移：{lr.intercept_}')
    # 训练集与测试集的得分差距较大，说明训练存在过拟合。（因为数据特征数目较多）
    print('训练集得分: {:.2f}'.format(lr.score(X_train, y_train)))
    print('测试集得分: {:.2f}'.format(lr.score(X_test, y_test)))


# 3) 岭回归（L2正则化）：替代标准的线性回归模型，可以控制模型的复杂度。
# 正则化：对模型做显式约束，以避免过拟合，
def compare_ridge_high_dimension_figure():
    """利用图形，展示岭回归在固定高维数据集的情况下，不同alpha值的效果"""
    X_train, X_test, y_train, y_test = load_train_test_extended_boston()

    # 线性回归
    lr = LinearRegression().fit(X_train, y_train)
    show_title("线性回归模型")
    print('训练集得分: {:.2f}'.format(lr.score(X_train, y_train)))
    print('测试集得分: {:.2f}'.format(lr.score(X_test, y_test)))

    plt.figure()
    plt.plot(lr.coef_, 's')
    plt.title('线性回归模型')
    plt.ylabel('线性回归模型的系数')
    plt.ylim(-50, 50)
    plt.suptitle("线性回归系数")

    # 岭回归：alpha 值的增加，降低训练集的精度，但是可以提高测试集的精度
    # alpha 值较小时，模型过拟合；alpha 值较大时，模型欠拟合
    show_title("岭回归模型")
    for alpha in [0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]:
        ridge = Ridge(alpha=alpha).fit(X_train, y_train)
        print('alpha: {}'.format(alpha))
        print('训练集得分: {:.2f}'.format(ridge.score(X_train, y_train)))
        print('测试集得分: {:.2f}'.format(ridge.score(X_test, y_test)))
        print('-' * 20)

        plt.figure()
        plt.plot(ridge.coef_, 's')
        plt.title('alpha={}'.format(alpha))
        plt.ylabel('岭回归模型的系数')
        plt.ylim(-50, 50)
        plt.suptitle("图2-12：不同alpha值的岭回归的系数比较")
        pass


def compare_ridge_high_dimension_coef_():
    """利用得分曲线，展示岭回归在固定高维数据集的情况下，不同alpha值的效果"""
    X_train, X_test, y_train, y_test = load_train_test_extended_boston()

    alpha_range = [pow(10, (alpha / 10)) for alpha in range(-50, 1)]
    train_score, test_score = [], []
    for alpha in alpha_range:
        ridge = Ridge(alpha=alpha).fit(X_train, y_train)
        train_score.append(ridge.score(X_train, y_train))
        test_score.append(ridge.score(X_test, y_test))

    plt.figure()
    plt.plot(alpha_range, train_score, label='训练集得分')
    plt.plot(alpha_range, test_score, label='测试集得分')
    plt.xlabel("alpha")
    plt.ylabel("得分")
    plt.legend()
    plt.suptitle("图2-12：不同alpha值的岭回归的系数曲线图")


# 岭回归和线性回归在Boston房价数据集上的学习曲线
def compare_ridge_fix_alpha():
    """利用学习曲线，展示岭回归在固定alpha值的情况下，不同数据量的的效果"""
    # 当数据集合中少于40个数据点时，线性回归学习不到任何有价值的信息；
    # 随着模型可用的数据越来越多，两个模型的性能都在提升
    # 当模型中可用的数据足够多时，两个模型的性能完全相同，即正则化不再重要
    mglearn.plots.plot_ridge_n_samples()
    plt.suptitle("图2-13：岭回归和线性回归在Boston房价数据集上的学习曲线")


# 4) Lasso 回归（L1正则化）：完全抑制某些特征（参数为0），自动化的特征选择
def compare_lasso_high_dimension_figure():
    """利用图形，展示Lasso回归在固定高维数据集的情况下，不同alpha值的效果"""
    X_train, X_test, y_train, y_test = load_train_test_extended_boston()

    # 线性回归
    lr = LinearRegression().fit(X_train, y_train)
    show_title("线性回归模型")
    print('训练集得分: {:.2f}'.format(lr.score(X_train, y_train)))
    print('测试集得分: {:.2f}'.format(lr.score(X_test, y_test)))
    print('=' * 20)
    plt.figure()
    plt.plot(lr.coef_, 's')
    plt.title('-- 线性回归模型 --')
    plt.ylabel('线性回归模型的系数')
    plt.ylim(-50, 50)
    plt.suptitle("线性回归系数")

    # Lasso 回归, alpha值较小时，模型过拟合
    for alpha in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1]:
        lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_train, y_train)
        print('alpha: {}'.format(alpha))
        print('训练集得分: {:.2f}'.format(lasso.score(X_train, y_train)))
        print('测试集得分: {:.2f}'.format(lasso.score(X_test, y_test)))
        print('有用的特征: {}'.format(np.sum(lasso.coef_ != 0)))
        # print(lasso)
        print('-' * 20)
        plt.figure()
        plt.plot(lasso.coef_, 's')
        plt.title('Lasso alpha={}'.format(alpha))
        plt.ylabel('Lasso回归模型的系数')
        plt.ylim(-50, 50)
        plt.suptitle("图2-14：不同alpha值的Lasso回归的系数曲线图")
        pass


def compare_lasso_high_dimension_coef_():
    """利用得分曲线，展示Lasso回归在固定高维数据集的情况下，不同alpha值的效果"""
    X_train, X_test, y_train, y_test = load_train_test_extended_boston()

    alpha_range = [pow(10, (alpha / 10)) for alpha in range(-50, -20)]
    train_score, test_score = [], []
    for alpha in alpha_range:
        lasso = Lasso(alpha=alpha, max_iter=100000).fit(X_train, y_train)
        train_score.append(lasso.score(X_train, y_train))
        test_score.append(lasso.score(X_test, y_test))

    plt.plot(alpha_range, train_score, label='训练集得分')
    plt.plot(alpha_range, test_score, label='测试集得分')
    plt.xlabel("alpha")
    plt.ylabel("score")
    plt.legend()
    plt.suptitle("图2-14：不同alpha值的Lasso回归的系数曲线图")


def compare_elastic_high_dimension_coef_():
    """利用得分曲线，展示Lasso回归和Ridge回归结合在一起，在固定高维数据集的情况下，不同alpha值的效果"""
    # 运行时间比较长
    X_train, X_test, y_train, y_test = load_train_test_extended_boston()

    alpha_range = [pow(10, (alpha / 10)) for alpha in range(-50, 0, 3)]
    lasso_train_score, lasso_test_score = [], []
    ridge_train_score, ridge_test_score = [], []
    elastic_train_score, elastic_test_score = [], []
    fix_elastic_train_score, fix_elastic_test_score = [], []
    for alpha in alpha_range:
        lasso = Lasso(alpha=alpha, max_iter=100000).fit(X_train, y_train)
        lasso_train_score.append(lasso.score(X_train, y_train))
        lasso_test_score.append(lasso.score(X_test, y_test))

        ridge = Ridge(alpha=alpha).fit(X_train, y_train)
        ridge_train_score.append(ridge.score(X_train, y_train))
        ridge_test_score.append(ridge.score(X_test, y_test))

        elastic = ElasticNet(alpha=alpha, l1_ratio=alpha).fit(X_train, y_train)
        elastic_train_score.append(elastic.score(X_train, y_train))
        elastic_test_score.append(elastic.score(X_test, y_test))

        # 将最好的L1正则化系数固定，变动L2系数
        elastic = ElasticNet(alpha=alpha, l1_ratio=0.005).fit(X_train, y_train)
        fix_elastic_train_score.append(elastic.score(X_train, y_train))
        fix_elastic_test_score.append(elastic.score(X_test, y_test))

    plt.plot(alpha_range, lasso_train_score, label='lasso 训练集得分')
    plt.plot(alpha_range, ridge_train_score, label='ridge 训练集得分')
    plt.plot(alpha_range, elastic_train_score, label='elastic 训练集得分')
    plt.plot(alpha_range, fix_elastic_train_score, label='fix elastic 训练集得分')
    plt.plot(alpha_range, lasso_test_score, label='lasso 测试集得分')
    plt.plot(alpha_range, ridge_test_score, label='ridge 测试集得分')
    plt.plot(alpha_range, elastic_test_score, label='elastic 测试集得分')
    plt.plot(alpha_range, fix_elastic_test_score, label='fix elastic 测试集得分')
    plt.legend(ncol=4, loc=(0, 1))
    plt.xlabel("alpha")
    plt.ylabel("score")
    plt.suptitle("不同alpha值的四种回归的系数曲线图")


# 5) 用于分类的线性模型
# 默认使用 L2 正则化
# 线性支持向量分类（Linear Support Vector Classification，LSVC）
def compare_linear_classification_figure():
    X, y = make_my_forge(n_samples=30)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # 两个模型得到了相似的决策边界，
    # 都有两个点的分类是错误的，
    # 都默认使用了L2正则化, C 是正则化参数
    for model, ax in zip([LinearSVC(C=0.1), LogisticRegression(C=0.1)], axes):
        clf = model.fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title('{}'.format(clf.__class__.__name__))
        ax.set_xlabel('特征 0')
        ax.set_ylabel('特征 1')

    axes[0].legend()
    plt.suptitle("图2-15：线性SVM和Logistic回归在forge数据集上的决策边界（均为默认参数）")


def compare_lsvc_figure():
    # 决定正则化强度的权衡参数叫做C。C值越大，正则化越弱。
    # 较小的C值让算法尽量适应“大多数”数据点
    # 较大的C值更强调每个数据点都分类正确的重要性
    mglearn.plots.plot_linear_svc_regularization()
    plt.suptitle("图2-16：不同C值的线性SVM在forge数据集上的决策边界")


# Logistic回归(Logistic Regression)
# 线性SVM和不同C值的Logistic回归在cancer数据集上学到的系数
def compare_logistic_classification_figure():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, stratify=cancer.target, random_state=seed)

    linear_svc = LinearSVC(max_iter=10000)
    linear_svc.fit(X_train, y_train)
    show_title("线性支持向量分类模型")
    print('训练集得分: {:.3f}'.format(linear_svc.score(X_train, y_train)))
    print('测试集得分: {:.3f}'.format(linear_svc.score(X_test, y_test)))
    print('*' * 20)
    plt.figure()
    plt.plot(linear_svc.coef_.T, 's', label='max_iter=10000')
    plt.xlabel('系数的索引')
    plt.ylabel('系数的量级')
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.ylim(-2.5, 2.5)
    plt.legend()
    plt.suptitle("线性SVM在cancer数据集上学到的系数")

    from sklearn.linear_model import LogisticRegression
    for C in [0.01, 0.1, 1, 10, 100]:
        # 训练集和测试集的性能非常接近，因此模型有可能欠拟合
        # 更复杂的模型性能更好，但是C值过大后也会发生过拟合问题
        logistic_regression = LogisticRegression(C=C, solver='lbfgs', max_iter=10000)
        logistic_regression.fit(X_train, y_train)
        show_title(f'Logistic Regression C={C} ')
        print('训练集得分: {:.3f}'.format(logistic_regression.score(X_train, y_train)))
        print('测试集得分: {:.3f}'.format(logistic_regression.score(X_test, y_test)))
        print('-' * 20)
        plt.figure()
        plt.plot(logistic_regression.coef_.T, 'o', label='C= {}'.format(C))
        plt.xlabel('系数的索引')
        plt.ylabel('系数的量级')
        plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
        plt.hlines(0, 0, cancer.data.shape[1])
        plt.ylim(-2.5, 2.5)
        plt.legend()
        plt.suptitle(f"图2-17：C={C}的Logistic回归在cancer数据集上学到的系数")
        pass


# 线性SVM和不同C值的L1 正则化的Logistic回归在cancer数据集上学到的系数
def compare_logistic_classification_l1_figure():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, stratify=cancer.target, random_state=seed)

    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        logistic_regression = LogisticRegression(penalty='l1', C=C, solver='liblinear', max_iter=10000)
        logistic_regression.fit(X_train, y_train)
        show_title(f'Logistic Regression C={C} ')
        print('训练集得分: {:.3f}'.format(logistic_regression.score(X_train, y_train)))
        print('测试集得分: {:.3f}'.format(logistic_regression.score(X_test, y_test)))
        print('-' * 20)
        plt.figure()
        plt.plot(logistic_regression.coef_.T, 'o', label=f'C= {C}')
        plt.xlabel('系数的索引')
        plt.ylabel('系数的量级')
        plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
        plt.hlines(0, 0, cancer.data.shape[1])
        plt.ylim()
        # plt.ylim(-5, 5)
        plt.legend()
        plt.suptitle(f"图2-18：C={C}的L1正则化的Logistic回归在cancer数据集上学到的系数")
        pass


# 图2-18：对于不同的C值，L1正则化的Logistic回归在Cancer数据集上学到的系数
def logistic_lasso_classification():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, stratify=cancer.target, random_state=seed)

    for C, marker in zip([0.001, 0.01, 0.1, 1, 10, 100], ['o', '^', 'v', 'h', '8', '+']):
        lr_l1 = LogisticRegression(penalty='l1', C=C, solver='liblinear', max_iter=10000)
        lr_l1.fit(X_train, y_train)
        show_title(f'Logistic回归模型的 C={C} ')
        print('L1正则化模型的训练集精度 {:.2f}'.format(lr_l1.score(X_train, y_train)))
        print('L1正则化模型的测试集精度 {:.2f}'.format(lr_l1.score(X_test, y_test)))
        print('-----')
        plt.plot(lr_l1.coef_.T, marker, label='C={:.3f}'.format(C))

    plt.xlabel('系数的索引')
    plt.ylabel('系数的量级')
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.ylim(-5, 5)
    plt.legend(loc=3)
    plt.suptitle("图2-18：对于不同的C值，L1正则化的Logistic回归在Cancer数据集上学到的系数")


# 6) 用于多分类的线性模型
def show_three_classes_dataset():
    # 从高斯分布中采样得到的三类数据，每个数据点具有两个特征
    # centers表示数据有几个中心，默认是3个中心
    X, y = make_blobs(random_state=seed, centers=3)
    print('数据的形状: ', X.shape)
    print('类别的形状: ', y.shape)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel('特征 0')
    plt.ylabel('特征 1')
    plt.legend(['类别 0', '类别 1', '类别 2'])
    plt.legend()
    plt.suptitle("图2-19：包含3个类别的二维简单数据集")


# “一对其余”分类器学到的二分类决策边界和得到的多分类决策边界
def linear_model_multi_classification():
    # 每次划分都是“一对其余”。因此到5个类别后，被包围的中间类别就无法正确分类了。
    CLASS_NUMBER = 3
    X, y = make_blobs(n_samples=100, random_state=seed, centers=CLASS_NUMBER)
    print('数据的形状: ', X.shape)
    print('类别的形状: ', y.shape)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

    linear_svm = LinearSVC(max_iter=10000)
    linear_svm.fit(X, y)
    print('系数的形状: ', linear_svm.coef_.shape)
    print('偏移的形状: ', linear_svm.intercept_.shape)

    line = np.linspace(-15, 15)
    for coef, intercept, color in zip(
            linear_svm.coef_, linear_svm.intercept_,
            ['blue', 'red', 'green', 'yellow', 'purple', 'black']):
        plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

    plt.xlim(-10, 8)
    plt.ylim(-10, 15)
    plt.xlabel('特征 0')
    plt.ylabel('特征 1')
    legend = [f'Class {x}' for x in range(CLASS_NUMBER)]
    legend = legend + [f'Line Class {x}' for x in range(CLASS_NUMBER)]
    plt.legend(legend, loc=(1.01, 0.3))
    plt.suptitle("图2-20：三个“一对其余”分类器学到的二分类决策边界")

    # 二维空间中所有区域的预测结果
    # 对于测试点的类别划分：测试点最接近的那条线对应的类别
    # 通过图形可以发现分类的区域不太合理
    plt.figure()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    # for coef, intercept, color in zip(
    #         linear_svm.coef_, linear_svm.intercept_, ['blue', 'red', 'green', 'yellow', 'purple', 'black']):
    #     plt.plot(line, -(line * coef[0] + intercept) / coef[1], c = color)

    plt.xlim(-10, 8)
    plt.ylim(-10, 15)
    plt.xlabel('特征 0')
    plt.ylabel('特征 1')
    legend = ['Class {}'.format(x) for x in range(CLASS_NUMBER)]
    plt.legend(legend, loc=(1.01, 0.3))
    mglearn.plots.plot_2d_classification(
            linear_svm, X, fill=True, alpha=.7, cm=mglearn.plot_helpers.cm_cycle)
    plt.suptitle("图2-21：三个“一对其余”分类器得到的多分类决策边界")


if __name__ == "__main__":
    # 1）用于回归的线性模型
    # linear_model_regression()

    # 2）线性回归，又叫最小二乘法（Least Squares Method，LSM），
    # least_squares()

    # 最小二乘法应用于高维数据
    # least_squares_high_dimension()

    # 3) 岭回归（L2正则化）：替代标准的线性回归模型，可以控制模型的复杂度。
    # 线性模型和不同alpha值的岭回归的系数比较
    # compare_ridge_high_dimension_figure()

    # 不同alpha值的岭回归的系数曲线图
    # compare_ridge_high_dimension_coef_()

    # 岭回归和线性回归在Boston房价数据集上的学习曲线
    # compare_ridge_fix_alpha()

    # 4) Lasso 回归（L1正则化）：完全抑制某些特征（参数为0），自动化的特征选择
    # 线性模型和不同alpha值的Lasso回归的系数比较
    # compare_lasso_high_dimension_figure()

    # 不同alpha值的Lasso回归的系数曲线图
    # compare_lasso_high_dimension_coef_()

    # 不同alpha值的四种回归的系数曲线图
    # compare_elastic_high_dimension_coef_()

    # 5) 用于分类的线性模型
    # 图2-15：线性SVM和Logistic回归在forge数据集上的决策边界（均为默认参数）
    # compare_linear_classification_figure()

    # 图2-16：不同C值的线性SVM在forge数据集上的决策边界
    # compare_lsvc_figure()

    # 线性SVM和不同C值的Logistic回归在cancer数据集上学到的系数
    # compare_logistic_classification_figure()

    # 线性SVM和不同C值的L1 正则化的Logistic回归在cancer数据集上学到的系数
    # compare_logistic_classification_l1_figure()

    # 图2-18：对于不同的C值，L1正则化的Logistic回归在Cancer数据集上学到的系数
    # logistic_lasso_classification()

    # 6) 用于多分类的线性模型
    # 包含3个类别的二维简单数据集
    # show_three_classes_dataset()

    # “一对其余”分类器学到的二分类决策边界和得到的多分类决策边界
    # linear_model_multi_classification()

    from tools import beep_end
    from tools import show_figures

    beep_end()
    show_figures()
