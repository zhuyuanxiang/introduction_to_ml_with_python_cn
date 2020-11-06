# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C08_my_transformer.py
@Version    :   v0.1
@Time       :   2019-10-14 18:22
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Ch08，P278
@Desc       :   全书总结。建立自己的分类器。
"""
from tools import *

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision=3, suppress=True, threshold=np.inf, linewidth=200)

import unittest
from sklearn.base import BaseEstimator, TransformerMixin


class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, first_parameter=1, second_parameter=2):
        self.first_parameter = first_parameter
        self.second_parameter = second_parameter
        pass

    def fit(self, X, y=None):
        # fit应该只接受X和y作为参数，即使模型是无监督的，也需要y参数。
        # 下面是模型拟合的代码
        return self

    def transform(self, X):
        # transform只接受X作为参数
        # 对X应用某种变换
        X_transformed = X + 1
        return X_transformed


class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test(self):
        self.assertEqual(True, True)
        self.fail()


if __name__ == "__main__":
    beep_end()
    show_figures()
