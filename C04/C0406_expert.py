# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C0406_expert.py
@Version    :   v0.1
@Time       :   2019-10-10 11:00
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec04
@Desc       :   
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)


# 4.6. 利用专家知识
def load_original_citibank_data():
    # 显式注册转换器，就不会输出"warning"了。
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    city_bike = mglearn.datasets.load_citibike()

    # 将city_bike数据变换成二维数组
    # X = city_bike.index.values.reshape(-1, 1)

    # 将city_bike数据变换成POSIX时间格式的二维数组
    X = city_bike.index.astype(np.int).values.reshape(-1, 1)
    y = city_bike.values

    number_title = "Citi Bike的原始数据及转换成POSIX时间格式的数据"
    print('\n', '-' * 5, number_title, '-' * 5)

    print('City Bike data:\n{}'.format(city_bike.head()))
    print('X=\n{}'.format(X[:5]))
    print('y=\n{}'.format(y[:5]))

    # 整个月租车数量的可视化
    plt.figure(figsize = (10, 5))
    xticks = pd.date_range(start = city_bike.index.min(), end = city_bike.index.max(), freq = 'D')
    plt.xticks(xticks, xticks.strftime('%a %m-%d'), rotation = 90, ha = 'left')
    plt.plot(city_bike, linewidth = 1)
    plt.xlabel('日期')
    plt.ylabel('出租')
    plt.suptitle("图4-12：对于选定的 Citi Bike站点，自行车出租数量随着时间的变化")


def eval_on_features(feature_values, target_values, regressor):
    """对给定的特征集上的回归进行评估和作图的函数"""
    # 使用前184个数据点用于训练，剩余的数据点用于测试
    n_train = 184
    # 将给定的特征划分为训练集和测试集
    X_train, X_test = feature_values[:n_train], feature_values[n_train:]
    y_train, y_test = target_values[:n_train], target_values[n_train:]

    regressor.fit(X_train, y_train)
    print("{}'s test-set R^2: {:.2f}".format(type(regressor).__name__, regressor.score(X_test, y_test)))

    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    plt.figure(figsize = (10, 3))
    xticks = pd.date_range(start = feature_values.min(), end = feature_values.max(), freq = 'D')
    plt.xticks(range(0, len(feature_values), 8), xticks.strftime('%a %m-%d'), rotation = 90, ha = 'left')
    plt.plot(range(n_train), y_train, label = 'train')
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label = 'test')
    plt.plot(range(n_train), y_pred_train, '--', label = 'prediction train')
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label = 'prediction test')
    plt.legend(loc = (1.01, 0))
    plt.xlabel('Date')
    plt.ylabel('Rentals')


def feature_POSIX():
    # 显式注册转换器，就不会输出"warning"了。
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    city_bike = mglearn.datasets.load_citibike()
    # 将city_bike数据变换成POSIX时间格式的二维数组
    X = city_bike.index.astype(np.int).values.reshape(-1, 1)
    y = city_bike.values

    # 第一次预测，没有任何先验知识，得到的预测是一条直线，而不是周期线
    number_title = "仅使用POSIX时间做出的预测"
    print('\n', '-' * 5, number_title, '-' * 5)

    print('City Bike data:\n{}'.format(city_bike.head()))
    print('X=\n{}'.format(X[:5]))
    print('y=\n{}'.format(y[:5]))

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression

    # “随机森林”的拟合效果最好，但是预测效果都不好
    for clf, title in [(RandomForestRegressor(n_estimators = 100, random_state = 0), "随机森林"),
                       (Ridge(), "岭回归"),
                       (Lasso(), "拉索回归"),
                       (LinearRegression(), "线性回归"),
                       (LogisticRegression(solver = 'lbfgs', multi_class = 'auto'), "Logistic回归")]:
        eval_on_features(X, y, clf)
        plt.suptitle("图4-13：{}仅使用POSIX时间做出的预测".format(title))
        pass
    pass


def feature_POSIX_everytime():
    # 显式注册转换器，就不会输出"warning"了。
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    city_bike = mglearn.datasets.load_citibike()
    # 将city_bike数据变换成POSIX时间格式的二维数组
    X = city_bike.index.astype(np.int).values.reshape(-1, 1)
    y = city_bike.values

    # 第二次预测，使用每天的时刻作为先验知识，得到的预测要优于没有先验知识
    X_hour = city_bike.index.hour.values.reshape(-1, 1)
    number_title = "使用每天的时刻作为先验知识做出的预测"
    print('\n', '-' * 5, number_title, '-' * 5)

    print('City Bike data:\n{}'.format(city_bike.head()))
    print('X_hour=\n{}'.format(X_hour[:5]))
    print('y=\n{}'.format(y[:5]))

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    # “随机森林”的拟合效果最好，但是预测效果都不好
    for clf, title in [(RandomForestRegressor(n_estimators = 100, random_state = 0), "随机森林"),
                       (Ridge(), "岭回归"),
                       (Lasso(), "拉索回归"),
                       (LinearRegression(), "线性回归"),
                       (LogisticRegression(solver = 'lbfgs', max_iter = 10000, multi_class = 'auto'), "Logistic回归")]:
        eval_on_features(X_hour, y, clf)
        plt.suptitle("图4-13：{}仅使用每天的时刻做出的预测".format(title))
        pass
    pass


def feature_POSIX_weekday_everytime():
    # 显式注册转换器，就不会输出"warning"了。
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    city_bike = mglearn.datasets.load_citibike()
    # 将city_bike数据变换成POSIX时间格式的二维数组
    X = city_bike.index.astype(np.int).values.reshape(-1, 1)
    y = city_bike.values

    # 第二次预测，使用每天的时刻作为先验知识，得到的预测要优于没有先验知识
    X_hour = city_bike.index.hour.values.reshape(-1, 1)

    # 第三次预测，使用周期为星期几为先验知识
    X_hour_week = np.hstack([city_bike.index.dayofweek.values.reshape(-1, 1), X_hour])
    number_title = "使用周期为星期几和每天的时刻作为先验知识做出的预测"
    print('\n', '-' * 5, number_title, '-' * 5)

    print('City Bike data:\n{}'.format(city_bike.head()))
    print('X_hour_week=\n{}'.format(X_hour_week[:5]))
    print('y=\n{}'.format(y[:5]))

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    for clf, title in [(RandomForestRegressor(n_estimators = 100, random_state = 0), "随机森林"),
                       (Ridge(), "岭回归"),
                       (Lasso(), "拉索回归"),
                       (LinearRegression(), "线性回归"),
                       (LogisticRegression(solver = 'lbfgs', max_iter = 10000, multi_class = 'auto'), "Logistic回归")]:
        eval_on_features(X_hour_week, y, clf)
        plt.suptitle("图4-13：{}使用一周的星期几和每天的时刻两个特征做出的预测".format(title))
        pass
    pass
    # 换成线性回归模型，使用周期为星期几为先验知识，效果差（因为使用整数编码星期几和时间，使数据被解释为连续变量，需要解释成分类变量）


def feature_POSIX_weekday_everytime_one_hot():
    # 显式注册转换器，就不会输出"warning"了。
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    city_bike = mglearn.datasets.load_citibike()
    # 将city_bike数据变换成POSIX时间格式的二维数组
    X = city_bike.index.astype(np.int).values.reshape(-1, 1)
    y = city_bike.values

    # 第二次预测，使用周期为天的先验知识，得到的预测要优于没有先验知识
    X_hour = city_bike.index.hour.values.reshape(-1, 1)

    # 第三次预测，使用周期为星期几为先验知识
    X_hour_week = np.hstack([city_bike.index.dayofweek.values.reshape(-1, 1), X_hour])

    # 使用 OneHotEncoder 将整数变换为分类变量
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(categories = 'auto')
    X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
    number_title = "使用One-Hot编码过的一周的星期几和每天的时刻作为先验知识做出的预测"
    print('\n', '-' * 5, number_title, '-' * 5)

    print('City Bike data:\n{}'.format(city_bike.head()))
    print('X_hour_week_onehot=\n{}'.format(X_hour_week_onehot[:5]))
    print('y=\n{}'.format(y[:5]))

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    for clf, title in [(RandomForestRegressor(n_estimators = 100, random_state = 0), "随机森林"),
                       (Ridge(), "岭回归"),
                       (Lasso(), "拉索回归"),
                       (LinearRegression(), "线性回归"),
                       (LogisticRegression(solver = 'lbfgs', max_iter = 10000, multi_class = 'auto'), "Logistic回归")]:
        eval_on_features(X_hour_week_onehot, y, clf)
        plt.suptitle("图4-13：{}使用One-Hot编码过的一周的星期几和每天的时刻两个特征做出的预测".format(title))
        pass
    pass


def feature_POSIX_weekday_everytime_one_hot_poly():
    # 显式注册转换器，就不会输出"warning"了。
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    city_bike = mglearn.datasets.load_citibike()
    # 将city_bike数据变换成POSIX时间格式的二维数组
    X = city_bike.index.astype(np.int).values.reshape(-1, 1)
    y = city_bike.values

    # 第二次预测，使用周期为天的先验知识，得到的预测要优于没有先验知识
    X_hour = city_bike.index.hour.values.reshape(-1, 1)

    # 第三次预测，使用周期为星期几为先验知识
    X_hour_week = np.hstack([city_bike.index.dayofweek.values.reshape(-1, 1), X_hour])

    # 使用 OneHotEncoder 将整数变换为分类变量
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(categories = 'auto')
    X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()

    # 使用原始特征的多项式来扩展连续特征，本质就是将特征向高维空间映射
    from sklearn.preprocessing import PolynomialFeatures

    poly_transformer = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)
    X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
    number_title = "将 One-Hot 编码的特征映射为多项式特征作为先验知识做出的预测"
    print('\n', '-' * 5, number_title, '-' * 5)

    import random
    rand_number_list = random.sample(range(0, len(y)), 5)
    print('City Bike data:\n{}'.format(city_bike[rand_number_list]))
    print('X_hour_week_onehot_poly=\n{}'.format(X_hour_week_onehot_poly[rand_number_list]))
    print('y=\n{}'.format(y[rand_number_list]))

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    rfr = RandomForestRegressor(n_estimators = 100, random_state = 0)
    ridge = Ridge()
    lasso = Lasso()
    linear_r = LinearRegression()
    logistic_r = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')

    eval_on_features(X_hour_week_onehot_poly, y, rfr)
    plt.suptitle("RandomForestRegressor()使用星期几和每天的时刻两个特征的乘积做出的预测")
    # 随机森林没有系数可以输出
    # model_coefficient(rfr,poly_transformer)
    # plt.suptitle("RandomForestRegressor()使用星期几和每天的时刻两个特征的乘积学到的系数")

    eval_on_features(X_hour_week_onehot_poly, y, ridge)
    # （Test-set R^2: 0.85）岭回归，正则函数是系数的L2范数，这个拟合效果好
    plt.suptitle("图4-18：Ridge()使用星期几和每天的时刻两个特征的乘积做出的预测")
    model_coefficient(ridge, poly_transformer)
    plt.suptitle("图4-19：Ridge()使用星期几和每天的时刻两个特征的乘积学到的系数")

    eval_on_features(X_hour_week_onehot_poly, y, lasso)
    # （Test-set R^2: 0.30）正则函数是系数的L1范数，这个拟合效果不好（可能是因为稀疏化造成的）
    plt.suptitle("Lasso()使用星期几和每天的时刻两个特征的乘积做出的预测")
    model_coefficient(lasso, poly_transformer)
    plt.suptitle("Lasso()使用星期几和每天的时刻两个特征的乘积学到的系数")

    eval_on_features(X_hour_week_onehot_poly, y, linear_r)
    # （Test-set R^2: 0.84）线性模型就已经可以很好的拟合了
    plt.suptitle("LinearRegression()使用星期几和每天的时刻两个特征的乘积做出的预测")
    model_coefficient(linear_r, poly_transformer)
    plt.suptitle("LinearRegression()使用星期几和每天的时刻两个特征的乘积学到的系数")

    eval_on_features(X_hour_week_onehot_poly, y, logistic_r)
    # （Test-set R^2: 0.14）Logistic回归，这个拟合效果不好（可能是因为函数的挤压特性，导致尖锐的细节都被丢失了）
    plt.suptitle("LogisticRegression()使用星期几和每天的时刻两个特征的乘积做出的预测")
    # LogisticRegression的系数展示不正常
    # model_coefficient(logistic_r,poly_transformer)
    # plt.suptitle("LogisticRegression()使用星期几和每天的时刻两个特征的乘积学到的系数")


def model_coefficient(model, poly_transformer):
    # 通过分析交互项学到的系数知道为什么Ridge()和LinearRegression()学习效果较好，而Lasso()的学习效果较差
    # ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    hour = ['%02d:00' % i for i in range(0, 24, 3)]
    # [' 0:00', ' 3:00', ' 6:00', ' 9:00', '12:00', '15:00', '18:00', '21:00']
    # hour = ['{:2d}:00'.format(i) for i in range(0, 24, 3)]
    day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    features = day + hour

    features_poly = poly_transformer.get_feature_names(features)
    features_nonzero = np.array(features_poly)[model.coef_ != 0]
    coef_nonzero = model.coef_[model.coef_ != 0]

    plt.figure(figsize = (15, 2))
    plt.plot(coef_nonzero, 'o')
    plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation = 90)
    plt.xlabel('Feature name')
    plt.ylabel('Feature magnitude')


if __name__ == "__main__":
    # Citi Bike的原始数据及转换成POSIX时间格式的数据
    # load_original_citibank_data()

    # 仅使用POSIX时间做出的预测
    # feature_POSIX()

    # 使用每天的时刻作为先验知识做出的预测
    # feature_POSIX_everytime()

    # 使用周期为星期几和每天的时刻作为先验知识做出的预测
    # feature_POSIX_weekday_everytime()

    #使用One-Hot编码过的一周的星期几和每天的时刻作为先验知识做出的预测
    # feature_POSIX_weekday_everytime_one_hot()

    # 将 One-Hot 编码的特征映射为多项式特征作为先验知识做出的预测
    feature_POSIX_weekday_everytime_one_hot_poly()

    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
