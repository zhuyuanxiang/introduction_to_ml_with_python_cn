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
import numpy as np

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision=3, suppress=True, threshold=np.inf)

# 2.3.4. 朴素贝叶斯分类器
# BernoulliNB分类器计算每个类别中每个特征不为0的元素个数
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

from tools import beep_end
from tools import show_figures

beep_end()
show_figures()

