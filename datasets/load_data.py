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
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from config import seed


# ----------------------------------------------------------------------
def load_train_test_wave(n_samples=100):
    X, y = make_my_wave(n_samples=n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_test, X_train, y_test, y_train


def load_train_test_breast_cancer():
    cancer = sklearn.datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, stratify=cancer.target, random_state=seed)
    return X_test, X_train, y_test, y_train


def make_my_forge(centers=2, n_samples=30):
    # a carefully hand-designed dataset lol
    X, y = sklearn.datasets.make_blobs(centers=centers, random_state=seed, n_samples=n_samples)
    y[np.array([7, 27])] = 0
    mask = np.ones(len(X), dtype=np.bool)
    mask[np.array([0, 1, 5, 26])] = 0
    X, y = X[mask], y[mask]
    return X, y


def make_my_wave(n_samples=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(-3, 3, size=n_samples)
    y_no_noise = (np.sin(4 * x) + x)
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    return x.reshape(-1, 1), y


def load_train_test_extended_boston():
    X, y = load_my_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_test, X_train, y_test, y_train


def load_my_extended_boston():
    from sklearn.datasets import load_boston
    boston = load_boston()
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    return X, boston.target


def load_train_test_iris():
    iris = sklearn.datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, stratify=iris.target, random_state=seed)
    return X_test, X_train, y_test, y_train


def load_train_test_moons(n_samples=100, noise=0.25):
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed)
    return X_test, X_train, y_test, y_train
