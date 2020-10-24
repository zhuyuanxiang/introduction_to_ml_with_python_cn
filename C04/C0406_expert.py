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
from tools import *


# 4.6. 利用专家知识
def load_original_citibank_data():
    # 显式注册转换器，就不会输出"warning"了。
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    from mglearn.datasets import load_citibike
    citi_bike = load_citibike()

    # 将city_bike数据变换成二维数组
    # X = citi_bike.index.values.reshape(-1, 1)

    # 将city_bike数据变换成posix时间格式的二维数组
    X = citi_bike.index.astype(np.int).values.reshape(-1, 1)
    y = citi_bike.values

    show_title("Citi Bike的原始数据及转换成posix时间格式的数据")

    print('City Bike data:\n{}'.format(citi_bike.head()))
    print('X=\n{}'.format(X[:5]))
    print('y=\n{}'.format(y[:5]))

    # 整个月租车数量的可视化
    plt.figure(figsize=(10, 5))
    xticks = pd.date_range(start=citi_bike.index.min(), end=citi_bike.index.max(), freq='D')
    plt.xticks(xticks, xticks.strftime('%a %m-%d'), rotation=90, ha='left')
    plt.plot(citi_bike, linewidth=1)
    plt.xlabel('日期')
    plt.ylabel('出租')
    plt.suptitle("图4-12：对于选定的 Citi Bike站点，自行车出租数量随着时间的变化")


def eval_on_features(feature_values, target_values, regressor):
    """对给定的特征集上的回归进行评估和作图的函数"""
    # 显式注册转换器，就不会输出"warning"了。
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    # 使用前184个数据点用于训练，剩余的数据点用于测试
    n_train = 184
    # 将给定的特征划分为训练集和测试集
    X_train, X_test = feature_values[:n_train], feature_values[n_train:]
    y_train, y_test = target_values[:n_train], target_values[n_train:]

    regressor.fit(X_train, y_train)
    print("{}'s 训练集 R^2: {:.2f}".format(type(regressor).__name__, regressor.score(X_train, y_train)))
    print("{}'s 测试集 R^2: {:.2f}".format(type(regressor).__name__, regressor.score(X_test, y_test)))

    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    plt.figure(figsize=(10, 3))
    # xticks = pd.date_range(start=feature_values.min(), end=feature_values.max(), freq='D')
    # plt.xticks(range(0, len(feature_values), 8), xticks.strftime('%a %m-%d'), rotation=90, ha='left')
    xticks = range(1, 32)  # FIXME：不想再修改这段代码了，就简化为 31 天的周期，
    plt.xticks(range(0, len(feature_values), 8), xticks, rotation=90, ha='left')
    plt.plot(range(n_train), y_train, label='train')
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label='test')
    plt.plot(range(n_train), y_pred_train, '--', label='prediction train')
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label='prediction test')
    plt.legend(loc=(1.01, 0))
    plt.xlabel('Date')
    plt.ylabel('Rentals')


def train_data(X, y, number_title):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    # “随机森林”的拟合效果最好
    # “LogisticRegression”的效果不好，因为这个模型适用于分类问题，而不是回归问题
    for clf, title in [(RandomForestRegressor(n_estimators=100, random_state=seed), "随机森林"),
                       (Ridge(), "岭回归"),
                       (Lasso(), "拉索回归"),
                       (LinearRegression(), "线性回归"),
                       (LogisticRegression(solver='lbfgs', multi_class='auto'), "Logistic回归")]:
        show_title(title)
        eval_on_features(X, y, clf)
        plt.suptitle(number_title.format(title))
        pass


def load_train_data():
    from mglearn.datasets import load_citibike
    citi_bike = mglearn.datasets.load_citibike()
    y = citi_bike.values
    print('City Bike data:\n{}'.format(citi_bike.head()))
    print('y=\n{}'.format(y[:5]))
    return citi_bike, y


def feature_posix():
    # 第一次预测，没有任何先验知识，得到的预测是一条直线，而不是周期线
    number_title = "仅使用posix时间做出的预测"
    show_title(number_title)
    citi_bike, y = load_train_data()
    # 将city_bike数据变换成posix时间格式的二维数组
    X = citi_bike.index.astype(np.int).values.reshape(-1, 1)
    print('X.shape={}'.format(X.shape))
    print('X=\n{}'.format(X[:5]))
    train_data(X, y, "图4-13：{}" + number_title)
    pass


def feature_posix_everytime():
    # 第二次预测，使用每天的时刻作为先验知识，得到的预测要优于没有先验知识
    number_title = "使用每天的时刻作为先验知识做出的预测"
    show_title(number_title)
    citi_bike, y = load_train_data()
    X_hour = citi_bike.index.hour.values.reshape(-1, 1)
    print('X_hour.shape={}'.format(X_hour.shape))
    print('X_hour=\n{}'.format(X_hour[:5]))
    train_data(X_hour, y, "图4-14：{}" + number_title)
    pass


def feature_posix_weekday_everytime():
    # 第二次预测，使用每天的时刻作为先验知识，得到的预测要优于没有先验知识
    # 第三次预测，增加周期为星期几为先验知识，随机森林的预测效果更优
    # 换成线性回归模型，使用周期为星期几为先验知识，效果差（因为使用整数编码星期几和时间，使数据被解释为连续变量，需要解释成分类变量）
    number_title = "使用周期为星期几和每天的时刻作为先验知识做出的预测"
    show_title(number_title)
    citi_bike, y = load_train_data()
    X_hour = citi_bike.index.hour.values.reshape(-1, 1)
    X_hour_week = np.hstack([citi_bike.index.dayofweek.values.reshape(-1, 1), X_hour])
    print('X_hour_week.shape={}'.format(X_hour.shape))
    print('X_hour_week=\n{}'.format(X_hour[:5]))
    train_data(X_hour_week, y, "图4-15/16：{}" + number_title)
    pass


def feature_posix_weekday_everytime_one_hot():
    # 第二次预测，使用每天的时刻作为先验知识，得到的预测要优于没有先验知识
    # 第三次预测，增加周期为星期几为先验知识，随机森林的预测效果更优
    # 换成线性回归模型，使用周期为星期几为先验知识，效果差（因为使用整数编码星期几和时间，使数据被解释为连续变量，需要解释成分类变量）
    number_title = "使用One-Hot编码周期为星期几和每天的时刻作为先验知识做出的预测"
    show_title(number_title)
    citi_bike, y = load_train_data()
    X_hour = citi_bike.index.hour.values.reshape(-1, 1)
    X_hour_week = np.hstack([citi_bike.index.dayofweek.values.reshape(-1, 1), X_hour])
    # 使用 OneHotEncoder 将整数变换为分类变量
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(categories='auto')
    X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
    print(f'X_hour_week_onehot.shape={X_hour_week_onehot.shape}')
    print(f'X_hour_week_onehot=\n{X_hour_week_onehot[:5]}')
    train_data(X_hour_week_onehot, y, "图4-17：{}" + number_title)
    pass


def feature_posix_weekday_everytime_one_hot_poly():
    # 第二次预测，使用每天的时刻作为先验知识，得到的预测要优于没有先验知识
    # 第三次预测，增加周期为星期几为先验知识，随机森林的预测效果更优
    # 换成线性回归模型，使用周期为星期几为先验知识，效果差（因为使用整数编码星期几和时间，使数据被解释为连续变量，需要解释成分类变量）
    number_title = "将 One-Hot 编码的特征映射为多项式特征作为先验知识做出的预测"
    show_title(number_title)
    city_bike, y = load_train_data()
    X_hour = city_bike.index.hour.values.reshape(-1, 1)
    X_hour_week = np.hstack([city_bike.index.dayofweek.values.reshape(-1, 1), X_hour])
    # 使用 OneHotEncoder 将整数变换为分类变量
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(categories='auto')
    X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
    # 使用原始特征的多项式来扩展连续特征，本质就是将特征向高维空间映射
    from sklearn.preprocessing import PolynomialFeatures
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
    print(f'X_hour_week_onehot_poly.shape={X_hour_week_onehot_poly.shape}')
    print(f'X_hour_week_onehot_poly=\n{X_hour_week_onehot_poly[:5]}')

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    # “随机森林”的拟合效果最好
    # “线性模型”的预测效果最好，因为构造的特征更符合线性模型的需要
    # “LogisticRegression”的效果很差，因为模型更适合分类问题
    for clf, title in [(RandomForestRegressor(n_estimators=100, random_state=seed), "随机森林"),
                       (Ridge(), "岭回归"),
                       (Lasso(), "拉索回归"),
                       (LinearRegression(), "线性回归"),
                       (LogisticRegression(solver='lbfgs', multi_class='auto'), "Logistic回归")]:
        eval_on_features(X_hour_week_onehot_poly, y, clf)
        plt.suptitle("图4-18：{}".format(title) + number_title)
        if type(clf) in [Ridge, Lasso, LinearRegression]:
            model_coefficient(clf, poly_transformer)
            plt.suptitle("图4-18：{}将 One-Hot 编码的特征映射为多项式特征作为先验知识学到的系数".format(title))
        pass


def model_coefficient(model, poly_transformer):
    # 通过分析交互项学到的系数知道为什么Ridge()和LinearRegression()学习效果较好，而Lasso()的学习效果较差
    # ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    hour = ['%02d:00' % i for i in range(0, 24, 3)]
    # [' 0:00', ' 3:00', ' 6:00', ' 9:00', '12:00', '15:00', '18:00', '21:00']
    # hour = ['{:2d}:00'.format(i) for i in range(0, 24, 3)]
    day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    features = day + hour

    features_poly = poly_transformer.get_feature_names(features)
    nonzero_idx = model.coef_ != 0
    features_nonzero = np.array(features_poly)[nonzero_idx]
    coef_nonzero = model.coef_[nonzero_idx]

    plt.figure(figsize=(15, 2))
    plt.plot(coef_nonzero, 'o')
    plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
    plt.xlabel('Feature name')
    plt.ylabel('Feature magnitude')


def main():
    # Citi Bike的原始数据及转换成posix时间格式的数据
    # load_original_citibank_data()
    # 仅使用posix时间做出的预测
    # feature_posix()
    # 使用每天的时刻作为先验知识做出的预测
    # feature_posix_everytime()
    # 使用周期为星期几和每天的时刻作为先验知识做出的预测
    # feature_posix_weekday_everytime()
    # 使用One-Hot编码过的一周的星期几和每天的时刻作为先验知识做出的预测
    feature_posix_weekday_everytime_one_hot()
    # 将 One-Hot 编码的特征映射为多项式特征作为先验知识做出的预测
    # feature_posix_weekday_everytime_one_hot_poly()
    pass


if __name__ == "__main__":
    main()
    beep_end()
    show_figures()
