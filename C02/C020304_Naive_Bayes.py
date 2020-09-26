# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C020304_Naive_Bayes.py
@Version    :   v0.1
@Time       :   2019-09-19 18:02
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec020304，P53
@Desc       :   监督学习算法。朴素贝叶斯分类器
"""

# Chap2 监督学习
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from config import *

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


# 2.3.4. 朴素贝叶斯分类器
# Scikit-Learn 中实现了三种朴素 Bayes 分类器：不同的分类器的区别在于对 $P(x_i|y)$ 的分布的假设
def train_GaussianNB():
    # GaussianNB：应用于任意的高维的连续数据，假设数据遵循 Gaussian 分布
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("GaussianNB：Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
    print(gnb.theta_)  # 每个类别每个特征的均值


def train_BernoulliNB():
    # BernoulliNB：假定输入数据为稀疏的二分类数据
    from sklearn.naive_bayes import BernoulliNB
    bnb = BernoulliNB()
    y_pred = bnb.fit(X_train, y_train).predict(X_test)
    print("BernoulliNB：Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))


def train_MultinomialNB():
    # MultinomialNB：假定输入数据为计数数据 ( 即每个特征代表某个对象的整数计数，例如：一个单词在句子中出现的次数 )
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB()
    y_pred = mnb.fit(X_train, y_train).predict(X_test)
    print("MultinomialNB：Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))


def train_ComplementNB():
    # ComplementNB：是MultinomialNB的自适应版本
    from sklearn.naive_bayes import ComplementNB
    mnb = ComplementNB()
    y_pred = mnb.fit(X_train, y_train).predict(X_test)
    print("ComplementNB：Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))


def train_CategoricalNB():
    # CategoricalNB：为类别化分布数据，每个特征都有自己的类别分布
    from sklearn.naive_bayes import CategoricalNB
    mnb = CategoricalNB()
    y_pred = mnb.fit(X_train, y_train).predict(X_test)
    print("CategoricalNB：Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))


def main():
    # BernoulliNB 分类器计算每个类别中每个特征不为0的元素个数
    X = np.array([[0, 1, 0, 1],
                  [1, 0, 1, 1],
                  [0, 0, 0, 1],
                  [1, 0, 1, 0],
                  [1, 1, 1, 0]])

    print('=' * 20)

    print('X:')
    print(X)

    # 根据y中类别选择X中的数据
    print('=' * 20)
    print('X[1]:', X[1])

    # 根据axis的数值，决定沿着X中第几维的数据进行计算
    print('-' * 20)
    print('X[1,:]:', X[1, :])

    print('-' * 20)
    print('X[:,1]:', X[:, 1])

    y = np.array([0, 0, 0, 1, 1])
    print('=' * 20)

    print('y=', y)

    print('-' * 20)

    print('X[y == 0]:')
    print(X[y == 0])

    print('-' * 20)
    print('X[y == 0].sum(axis = 0)', X[y == 0].sum(axis=0))

    print('-' * 20)
    print('X[y == 0].sum(axis = 1)', X[y == 0].sum(axis=1))

    # 没有第3维的数据
    # print('-' * 20)
    # print('X[y == 0].sum(axis = 2)', X[y == 0].sum(axis = 2))

    print('=' * 20)

    print('X[y == 1]:')
    print(X[y == 1])

    print('-' * 20)
    print('X[y == 1].sum(axis = 0)', X[y == 1].sum(axis=0))

    print('-' * 20)
    print('X[y == 1].sum(axis = 1)', X[y == 1].sum(axis=1))

    print('=' * 20)

    counts = {}
    for label in np.unique(y):
        # 对每个类别进行遍历
        # 计算（求和）每个特征中1的个数
        counts[label] = X[y == label].sum(axis=0)
        pass
    print('Feature counts:{}\t'.format(counts))


if __name__ == "__main__":
    train_GaussianNB()
    train_BernoulliNB()
    train_MultinomialNB()
    train_ComplementNB()
    train_CategoricalNB()

    from tools import beep_end
    from tools import show_figures

    beep_end()
    show_figures()
