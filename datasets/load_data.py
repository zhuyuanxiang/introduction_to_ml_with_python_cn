# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   load_data.py
@Version    :   v0.1
@Time       :   2020-08-28 12:19
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import sklearn
from sklearn.model_selection import train_test_split

from preamble import *


# ----------------------------------------------------------------------
def make_my_wave(n_samples=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(-3, 3, size=n_samples)
    y_no_noise = (np.sin(4 * x) + x)
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    return x.reshape(-1, 1), y


def load_train_test_wave(n_samples=100):
    X, y = make_my_wave(n_samples=n_samples)
    return train_test_split(X, y, random_state=0)


# ----------------------------------------------------------------------
def load_train_test_breast_cancer():
    cancer = sklearn.datasets.load_breast_cancer()
    return train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)


# ----------------------------------------------------------------------
def make_my_forge(centers=2, n_samples=30):
    # a carefully hand-designed dataset lol
    X, y = sklearn.datasets.make_blobs(centers=centers, random_state=4, n_samples=n_samples)
    y[np.array([7, 27])] = 0
    mask = np.ones(len(X), dtype=np.bool)
    mask[np.array([0, 1, 5, 26])] = 0
    return X[mask], y[mask]


# ----------------------------------------------------------------------
def load_my_extended_boston():
    boston = sklearn.datasets.load_boston()
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    return X, boston.target


def load_train_test_extended_boston():
    X, y = load_my_extended_boston()
    return train_test_split(X, y, random_state=0)


# ----------------------------------------------------------------------
def load_train_test_boston():
    boston = sklearn.datasets.load_boston()
    return train_test_split(boston.data, boston.target, random_state=0)


# ----------------------------------------------------------------------
def load_train_test_iris():
    iris = sklearn.datasets.load_iris()
    return train_test_split(iris.data, iris.target, stratify=iris.target, random_state=0)


# ----------------------------------------------------------------------
def load_train_test_moons(n_samples=100, noise=0.25):
    X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise, random_state=0)
    return train_test_split(X, y, stratify=y, random_state=0)


# ----------------------------------------------------------------------
def load_people():
    # 将文件 `lfw_home.zip` 解压到 `~/scikit-learn_data` 目录下
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
    return people


def load_train_test_faces():
    people = load_people()
    # 生成一个全0的mask矩阵
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        # 将每个人的前50条数据设置为1，方便取出
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    y_people = people.target[mask]
    # 将灰度值缩放到[0,1]之间，而不是[0,255]之间，可以得到更好的数据稳定性
    X_people = X_people / 255.
    return train_test_split(X_people, y_people, stratify=y_people, random_state=0)


# ----------------------------------------------------------------------
def load_adult_data():
    # adult.data 中没有 header，因此 header=None，使用 names 显式提供数据列的名称
    return pd.read_csv('../data/adult.data', header=None, index_col=False,
                       names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                              'marital-status', 'occupation', 'relationship', 'race', 'gender',
                              'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])


# ----------------------------------------------------------------------
def load_train_test_blobs():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(random_state=0)
    return train_test_split(X, y, random_state=0)
