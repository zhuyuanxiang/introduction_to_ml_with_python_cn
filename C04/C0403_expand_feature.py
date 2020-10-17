# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C0403_expand_feature.py
@Version    :   v0.1
@Time       :   2019-10-17 18:08
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec0403，P171
@Desc       :   数据表示与特征工程。扩展特征（交互特征和多项式特征）
"""
from pprint import pprint

from tools import *


# 4.3. 交互特征 与 多项式特征 这种特征工程学用于统计建模
# 使用线性回归模型 不仅可以学习偏移 还可以学习斜率
# 深入理解线性回归模型，才能理解分箱特征、One-Hot特征、多项式特征加入后为什么会产生这样的效果。
# 1）增加一个X特征，当然全局也就只有一个斜率
def linear_regression_binning_data():
    import random
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder
    # train_number = 100
    # test_number = 100
    # bin_number = 11

    for train_number in [500, 1000, 1500]:
        # 准备训练数据，有噪声的正弦波
        from mglearn.datasets import make_wave
        X_train, y_train = make_wave(n_samples=train_number)

        # 准备训练数据，无噪声的正弦波
        # X_train = np.linspace(-3, 3, train_number).reshape(-1, 1)
        # y_train = np.sin(X_train)

        fig, axes = plt.subplots(3, 3, figsize=(20, 10))
        plt.suptitle("图4-3：使用分箱特征和单一全局斜率的线性回归\n")
        for test_number, axs in zip([50, 100, 150], axes):
            # 准备测试数据，有噪声的正弦波
            from mglearn.datasets import make_wave
            X_test, y_test = make_wave(n_samples=test_number)
            # zip(*sorted(zip(X_test,y_test))) 中的 * 代表函数解包，就是将输出的参数打包成一个列表供 zip 使用
            X_test, y_test = (np.array(t) for t in zip(*sorted(zip(X_test, y_test))))

            # 准备测试数据，无噪声的正弦波
            # X_test = np.linspace(-3, 3, test_number).reshape(-1, 1)
            # y_test = np.sin(X_test)

            # 将X的值与箱子的值对应，即将连续值离散化
            for bin_number, ax in zip([6, 12, 18], axs):
                number_title = "{}个训练数据；{}个测试数据；{}个箱子".format(train_number, test_number, bin_number)
                show_title(number_title)

                bins = np.linspace(-3, 3, bin_number)
                # bins = np.linspace(X_train.min(), X_train.max(), bin_number)
                print("箱子的区间值：", bins)
                X_train_bin = np.digitize(X_train, bins=bins)

                # 将离散化的训练数据使用 OneHotEncoder 进行编码
                encoder = OneHotEncoder(sparse=False, categories='auto')
                encoder.fit(X_train_bin)
                X_train_one_hot = encoder.transform(X_train_bin)

                # 将X_test的值与箱子的值对应，即将连续值离散化
                X_test_bin = np.digitize(X_test, bins=bins)

                # 将离散化的测试数据使用 OneHotEncoder 进行编码
                # 必须使用训练数据集的编码器来编码，不能再训练一次
                X_test_one_hot = encoder.transform(X_test_bin)

                rand_train_number_list = random.sample(range(0, train_number), 5)
                X_train_combined = np.hstack([X_train, X_train_one_hot])

                print('-' * 50)
                print("将原始训练数据和 OneHotEncoder 进行编码的训练数据合并到一起")
                print("合并后的训练数据形状 =", X_train_combined.shape)
                print("五个合并后的数据=")
                print(X_train_combined[rand_train_number_list])

                # 合并时，注意测试数据与训练数据的对应关系。
                X_test_combined = np.hstack([X_test, X_test_one_hot])
                print('-' * 50)
                print("将原始测试数据和 OneHotEncoder 进行编码的测试数据合并到一起")
                print("合并后的测试数据形状 =", X_test_combined.shape)

                # 线性回归预测
                reg_lr = LinearRegression().fit(X_train_combined, y_train)
                print('-' * 50)
                print("使用线性回归预测的结果：「R^2」评价 = {}".format(reg_lr.score(X_test_combined, y_test)))

                # 画图
                # 划出分隔线
                for bin in bins:
                    ax.plot([bin, bin], [-3, -3], ':', c='k')
                    pass

                # 绘制数据点
                ax.plot(X_train[:, 0], y_train, 'o', c='b')
                ax.plot(X_test[:, 0], y_test, '^', c='r')
                ax.plot(X_test, reg_lr.predict(X_test_combined), label='分箱的线性回归')
                ax.legend(loc='best')
                if test_number == 150:
                    ax.set_xlabel('输入的特征')
                if bin_number == 6:
                    ax.set_ylabel('输出的回归')
                ax.set_title(number_title)


# 2）增加交互特征 或 乘积特征，就可以每个箱子都具有不同的斜率
# 利用One-Hot编码增加10个X特征，就可以学到10个斜率，不再箱子里面的X特征就被0给过滤了。
def linear_regression_add_new_feature_in_wave():
    train_number = 100
    test_number = 50
    bin_number = 12
    # 准备训练数据
    from mglearn.datasets import make_wave
    X_train, y_train = make_wave(n_samples=train_number)

    # 将X的值与箱子的值对应，即将连续值离散化
    bins = np.linspace(-3, 3, bin_number)
    x_train_binned = np.digitize(X_train, bins=bins)

    # 将离散特征使用 OneHotEncoder 进行编码
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False, categories='auto')
    encoder.fit(x_train_binned)
    X_train_one_hot = encoder.transform(x_train_binned)

    # 准备测试数据
    # X_test = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    X_test, y_test = make_wave(n_samples=test_number)
    X_test, y_test = (np.array(t) for t in zip(*sorted(zip(X_test, y_test))))

    # 将刻度使用 OneHotEncoder 进行编码
    X_test_one_hot = encoder.transform(np.digitize(X_test, bins=bins))

    import random
    rand_train_number_list = random.sample(range(0, train_number), 5)

    # 构造乘积特征，使One-Hot编码特征具有更加丰富的信息
    # 十个特征，斜率依然受到全部训练数据的影响，斜率方向是一致的
    number_title = "总共十个特征"
    show_title(number_title)
    X_train_product = np.hstack([X_train * X_train_one_hot])  # X_train*X_train_one_hot 不是矩阵乘，只是数组对应乘
    X_test_product = np.hstack([X_test * X_test_one_hot])

    print("X_train_product.shape=", X_train_product.shape)
    print("X_train_product=\n", X_train_product[rand_train_number_list])

    show_linear_regression(X_train, X_train_product, y_train, X_test, X_test_product, y_test, bins)
    plt.suptitle("图4-4：每个箱子具有不同斜率的线性回归--" + number_title)

    # 二十个特征，斜率不再受到全部训练数据的影响，斜率方向是不一致的
    number_title = "总共二十个特征"
    print('-' * 5, number_title, '-' * 5)
    X_train_product = np.hstack([X_train_one_hot, X_train * X_train_one_hot])  # X_train*X_train_one_hot 不是矩阵乘，只是数组对应乘
    X_test_product = np.hstack([X_test_one_hot, X_test * X_test_one_hot])
    print("X_train_product.shape=", X_train_product.shape)
    print("X_train_product=\n", X_train_product[rand_train_number_list])

    show_linear_regression(X_train, X_train_product, y_train, X_test, X_test_product, y_test, bins)
    plt.suptitle("图4-4：每个箱子具有不同斜率的线性回归--" + number_title)


def show_linear_regression(X_train, X_train_expand, y_train, X_test, X_test_expand, y_test, bins):
    # 线性回归预测，维度过高会造成过拟合
    from sklearn.linear_model import LinearRegression
    reg_lr = LinearRegression().fit(X_train_expand, y_train)
    y_predict = reg_lr.predict(X_test_expand)
    print('-' * 50)
    print("使用线性回归预测的结果：「R^2」评价 = {}".format(reg_lr.score(X_test_expand, y_test)))

    plt.figure()
    # 划出分隔线
    for bin in bins:
        plt.plot([bin, bin], [-3, -3], ':', c='k')
        pass
    # 绘制数据点
    plt.plot(X_train[:, 0], y_train, 'o', c='k')
    plt.plot(X_test, y_predict, label='分箱的线性回归')
    plt.xlabel('输入的特征')
    plt.ylabel('输出的回归')
    plt.legend(loc='best')


# 增加多项式特征
def linear_regression_add_polynomial_feature_in_wave():
    train_number = 100
    test_number = 100
    bin_number = 12
    # 准备训练数据
    from mglearn.datasets import make_wave
    X_train, y_train = make_wave(n_samples=train_number)

    # 将X的值与箱子的值对应，即将连续值离散化
    bins = np.linspace(-3, 3, bin_number)
    X_train_binned = np.digitize(X_train, bins=bins)

    # 将离散特征使用 OneHotEncoder 进行编码
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False, categories='auto')
    encoder.fit(X_train_binned)
    X_train_one_hot = encoder.transform(X_train_binned)

    # 准备测试数据
    X_test = np.linspace(-3, 3, test_number, endpoint=False).reshape(-1, 1)
    y_test = np.sin(X_test)

    # 将刻度使用 OneHotEncoder 进行编码
    X_test_one_hot = encoder.transform(np.digitize(X_test, bins=bins))

    # 使用原始特征的多项式来扩展连续特征，本质就是将特征向高维空间映射
    from sklearn.preprocessing import PolynomialFeatures

    import random
    rand_train_number_list = random.sample(range(0, train_number), 5)

    # include_bias，不包括偏置就是bin_number-1个特征，包括偏置就是bin_number个特征，偏置就是x0^0=1
    # poly_feature = PolynomialFeatures(degree = bin_number - 1)
    poly_feature = PolynomialFeatures(degree=bin_number - 1, include_bias=False)
    poly_feature.fit(X_train)
    X_train_poly = poly_feature.transform(X_train)
    X_test_poly = poly_feature.transform(X_test)

    print('-' * 20)
    print('X_train_poly.shape= {}'.format(X_train_poly.shape))
    print('X_train_poly=')
    print(X_train_poly[rand_train_number_list])
    print('Polynomial feature names=\n{}'.format(poly_feature.get_feature_names()))

    show_linear_regression(X_train, X_train_poly, y_train, X_test, X_test_poly, y_test, bins)
    plt.suptitle("图4-5：具有10次多项式特征的线性回归--没有One-Hot编码")

    # 加入One-Hot编码可以分段拟合多项式特征的线性回归
    # 这个对模型拟合并没有好处
    X_train_poly = np.hstack([X_train_one_hot, X_train_poly * X_train_one_hot])
    X_test_poly = np.hstack([X_test_one_hot, X_test_poly * X_test_one_hot])

    print('-' * 20)
    print('X_train_poly.shape= {}'.format(X_train_poly.shape))
    print('X_train_poly=')
    print(X_train_poly[rand_train_number_list])
    print('Polynomial feature names=\n{}'.format(poly_feature.get_feature_names()))

    show_linear_regression(X_train, X_train_poly, y_train, X_test, X_test_poly, y_test, bins)
    plt.suptitle("图4-5：具有10次多项式特征的线性回归--加入One-Hot编码")


def svm_in_wave():
    # 对比核 SVM 模型，调整 gamma 参数可以学到与多项式回归的复杂度类似的预测结果，而且还不需要进行显式的特征变换
    # 准备训练数据
    from mglearn.datasets import make_wave
    X, y = make_wave(n_samples=100)

    # 准备测试数据
    X_test = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    y_test = np.sin(X_test)

    from sklearn.svm import SVR

    # RBF核参数gamma，高斯核宽度的倒数，值越大，高斯分布越尖锐，方差越小
    for gamma in [0.1, 1, 3, 6, 10]:
        svr = SVR(gamma=gamma)
        svr.fit(X, y)
        print(f"SVR gamma={gamma} : ", svr.score(X_test, y_test))
        plt.plot(X_test, svr.predict(X_test), label='SVR gamma={}'.format(gamma))

    # 画图
    plt.plot(X[:, 0], y, 'o', c='k')
    plt.xlabel('Input feature')
    plt.ylabel('Regression output')
    plt.legend(loc='best')
    plt.suptitle("图4-6：对于RBF核的SVM，使用不同的gamma参数的对比\n" "不需要显式地特征变换就可以学习得到与多项式回归一样复杂的模型")


def add_new_feature_in_boston():
    # 继续学习多项式特征的应用，数据集为波士顿房价
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=seed)

    # 缩放数据
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # 提取多项式特征和交互特征，多项式特征的最高幂次数为2
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
    X_train_poly = poly.transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    show_title("为 Boston 数据集增加特征")
    print('原始训练数据的形状: {}'.format(X_train.shape))
    print('缩放过的原始训练数据的形状: {}'.format(X_train_scaled.shape))
    print('生成多项式特征和交互特征的缩放过后原始训练数据的形状: {}'.format(X_train_poly.shape))
    print('-' * 50)
    print('多项式特征的数量:{}'.format(len(poly.get_feature_names())))
    print('多项式特征的名称:{}'.format(poly.get_feature_names()))
    pprint('pprint.多项式特征的名称:{}'.format(poly.get_feature_names()), depth=13)

    # 对比缩放后的数据与增加了特征的数据上在线性模型 Ridge 上的精确度，变换特征后，精确度会得到提升
    from sklearn.linear_model import Ridge
    number_title = "Ridge训练不同特征数据的对比"
    show_title(number_title)
    ridge_scaled = Ridge().fit(X_train_scaled, y_train)
    print('缩放过的数据的评分: {:.3f}'.format(ridge_scaled.score(X_test_scaled, y_test)))
    ridge_poly = Ridge().fit(X_train_poly, y_train)
    print('生成多项式特征和交互特征的缩放过的数据的评分: {:.3f}'.format(ridge_poly.score(X_test_poly, y_test)))
    print("结论：增加交互特征后，精确度会提高")

    # 对比缩放后的数据与增加了特征的数据上在随机森林 RandomForest 上的精确度，变换特征后，性能反而会下降（可能是变换特征并非原始特征）
    from sklearn.ensemble import RandomForestRegressor
    number_title = "RandomForest 训练不同特征数据的对比"
    show_title(number_title)
    rf_scaled = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
    print('缩放过的数据的评分: {:.3f}'.format(rf_scaled.score(X_test_scaled, y_test)))
    rf_poly = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
    print('生成多项式特征和交互特征的缩放过的数据的评分: {:.3f}'.format(rf_poly.score(X_test_poly, y_test)))
    print("结论：增加交互特征后，精确度可能会降低（随机初始化，结果会不同）")

    from sklearn.ensemble import GradientBoostingRegressor
    number_title = "GradientBoostingRegressor 训练不同特征数据的对比"
    show_title(number_title)
    gbrt_scaled = GradientBoostingRegressor(n_estimators=100).fit(X_train_scaled, y_train)
    print('缩放过的数据的评分: {:.3f}'.format(gbrt_scaled.score(X_test_scaled, y_test)))
    gbrt_poly = GradientBoostingRegressor(n_estimators=100).fit(X_train_poly, y_train)
    print('生成多项式特征和交互特征的缩放过的数据的评分: {:.3f}'.format(gbrt_poly.score(X_test_poly, y_test)))
    print("结论：增加交互特征后，精确度可能会降低（随机初始化，结果会不同）")
    pass


def main():
    # 1）增加一个X特征，当然全局也就只有一个斜率
    # linear_regression_binning_data()
    # 2）增加交互特征 或 乘积特征，就可以每个箱子都具有不同的斜率
    # 利用One-Hot编码增加10个X特征，就可以学到10个斜率，不在箱子里面的X特征就被0给过滤了。
    # linear_regression_add_new_feature_in_wave()
    # 增加多项式特征
    # linear_regression_add_polynomial_feature_in_wave()
    # 使用核SVM模型来学习回归，可以得到非常平滑的拟合，并且不需要显式的特征变换。
    # svm_in_wave()
    # 利用Boston数据集，实际检验数据特征变换的效果
    add_new_feature_in_boston()
    pass


if __name__ == "__main__":
    main()
    beep_end()
    show_figures()
