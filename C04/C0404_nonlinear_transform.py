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
import matplotlib.pyplot as plt
import numpy as np

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)

# 4.4. 单变量非线性变换
def single_variable_nonlinear_transform():
    rnd = np.random.RandomState(0)
    X_org = rnd.normal(size = (1000, 3))
    w = rnd.normal(size = 3)

    # ToDo: 这个转换不理解，为什么先得到一个正态分布，再转换成泊松分布呢？
    X = rnd.poisson(10 * np.exp(X_org))
    y = np.dot(X_org, w)

    print('Number of feature appearances:\n{}'.format(np.bincount(X[:, 0])))

    bins = np.bincount(X[:, 0])
    plt.figure()
    plt.bar(range(len(bins)), bins)
    plt.xlabel('Value')
    plt.ylabel('Number of appearances')
    plt.suptitle("图4-7：X[:,0]特征取值的直方图")

    # 这样的泊松分布数据很难处理！
    # 尝试使用岭回归模型
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    print('-' * 20)
    number_title = "泊松分布数据使用岭回归模型来拟合"
    print('-' * 5, number_title, '-' * 5)
    print('Test score: {:.3f}'.format(Ridge().fit(X_train, y_train).score(X_test, y_test)))

    # 应用对数变换后，可能有用
    X_train_log = np.log(X_train + 1)
    X_test_log = np.log(X_test + 1)

    plt.figure()
    plt.hist(X_train_log[:, 0], bins = 25)
    plt.xlabel('Value')
    plt.ylabel('Number of appearances')
    plt.suptitle("图4-8：对X[:,0]特征取值进行对数变换后的直方图")

    # 再使用岭回归模型学习数据，拟合效果变好。（理解：将数据转换为高斯分布，因为岭回归就是基于L2正则的最小二乘法）
    print('-' * 20)
    print("应用对数变换后的泊松分布数据使用岭回归模型来拟合。")
    print('Test score: {:.3f}'.format(Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)))

    pass

if __name__ == "__main__":
    single_variable_nonlinear_transform()
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass 