# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C0404_nonlinear_transform.py
@Version    :   v0.1
@Time       :   2019-10-10 09:58
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec0404，P178
@Desc       :   数据表示与特征工程。单变量非线性变换。
"""
from tools import *


# 4.4. 单变量非线性变换
def single_variable_nonlinear_transform():
    rnd = np.random.RandomState(0)
    X_org = rnd.normal(size=(100000, 3))
    w = rnd.normal(size=3)

    # 因为泊松分布与正态分布之间的关系，因此使用正态分布建立泊松分布
    # 二项分布、泊松分布与正态分布之间的关系可以参考：
    # https://hongyitong.github.io/2016/11/13/%E4%BA%8C%E9%A1%B9%E5%88%86%E5%B8%83%E3%80%81%E6%B3%8A%E6%9D%BE%E5%88%86%E5%B8%83%E3%80%81%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83/
    X = rnd.poisson(10 * np.exp(X_org))
    y = np.dot(X_org, w)

    bins = np.bincount(X[:, 0])
    print('每个特征出现的次数：\n{}'.format(bins[:40]))

    plt.figure()
    plt.bar(range(len(bins)), bins)
    plt.xlabel('值')
    plt.ylabel('每个特征出现的次数')
    plt.suptitle("图4-7：X[:,0]特征取值的直方图")

    # 这样的泊松分布数据很难处理！
    # 尝试使用岭回归模型
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    show_title("泊松分布数据使用岭回归模型来拟合")
    print('测试集的得分 {:.3f}'.format(Ridge().fit(X_train, y_train).score(X_test, y_test)))

    # 应用对数变换有效（因为数据本身就是基于高斯分布使用指数函数生成的）
    X_train_log = np.log(X_train + 1)
    X_test_log = np.log(X_test + 1)

    plt.figure()
    plt.hist(X_train_log[:, 0], bins=25)
    plt.xlabel('值')
    plt.ylabel('特征表示的数目')
    plt.suptitle("图4-8：对X[:,0]特征取值进行对数变换后的直方图")

    # 再使用岭回归模型学习数据，拟合效果变好。（理解：将数据转换为高斯分布，因为岭回归就是基于L2正则的最小二乘法）
    # 所有的线性模型对于高斯分布的数据效果都比较好
    show_title("应用对数变换后的泊松分布数据使用岭回归模型来拟合")
    print('测试集的得分 {:.3f}'.format(Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)))

    pass


if __name__ == "__main__":
    single_variable_nonlinear_transform()
    beep_end()
    show_figures()
