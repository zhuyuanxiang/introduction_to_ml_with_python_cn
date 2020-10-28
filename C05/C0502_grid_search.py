# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C0502_grid_search.py
@Version    :   v0.1
@Time       :   2019-10-11 11:41
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec0502，P200
@Desc       :   模型评估与改进
"""
from sklearn.model_selection import train_test_split

from datasets.load_data import load_train_test_iris
from tools import *


# 5.2 网格搜索
# 从参数的所有组合中找出最优的参数设置。
# 5.2.1 简单的网格搜索
# 没有使用 Scikit-Learn 提供的函数，而是自己手工实现的
def simple_grid_search():
    X_train, X_test, y_train, y_test = load_train_test_iris()
    print("训练集合的大小：{}".format(X_train.shape[0]))
    print("测试集合的大小：{}".format(X_test.shape[0]))

    show_title("简单的网格搜索对 SVC 模型进行评分")
    best_score = 0
    best_parameters = {'C': 0, 'gamma': 0}
    from sklearn.svm import SVC
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
            svm = SVC(gamma=gamma, C=C)
            svm.fit(X_train, y_train)
            score = svm.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}
                show_subtitle("现在为止的最优参数")
                print("best_parameters =", best_parameters)
                print("best_score =", best_score)

    show_subtitle("搜索到的最佳模型")
    print("最佳参数：{}".format(best_parameters))
    print("最佳得分：{:.5f}".format(best_score))
    pass


# 5.2.2 参数过拟合的风险与验证集
def three_fold_split_data_distribution():
    # mglearn.plots.plot_threefold_split()
    plt.figure(figsize=(15, 3))
    axis = plt.gca()
    bars = axis.barh([0, 0, 0], [11.9, 2.9, 4.9], left=[0, 12, 15], height=0.4,
                     color=['white', 'grey', 'grey'], hatch="//", edgecolor='k',
                     align='edge')
    bars[2].set_hatch(r"")
    axis.set_frame_on(False)
    axis.set_ylim(-.4, .4)
    axis.set_xlim(-.1, 20.1)
    axis.set_xticks([6, 13.5, 17.5])
    axis.set_xticklabels(["训练集（Training Set）", "验证集（Validation Set）", "测试集（Test Set）"], fontdict={'fontsize': 10})
    axis.set_yticks(())
    axis.tick_params(length=0, labeltop=True, labelbottom=False)
    axis.text(6, -.1, "Model fitting", fontdict={'fontsize': 10}, horizontalalignment="center")
    axis.text(13.5, -.1, "Parameter selection", fontdict={'fontsize': 10}, horizontalalignment="center")
    axis.text(17.5, -.1, "Evaluation", fontdict={'fontsize': 10}, horizontalalignment="center")
    plt.suptitle("图5-5：对数据进行3折划分")
    pass


# 下面的例子可以看作自动修正参数的过程，如果通过测试集的评估来修正模型训练使用的参数，
# 那么最终评估模型泛化能力时会因为评估过程的反复修正而导致泛化能力被高估。
# 如果增加验证集来修正模型训练使用的参数，
# 那么最终评估模型泛化能力时，因为使用的是未知数据，所以得到的泛化性能评估更加可信。
# 利用测试集修正模型参数导致的误差称作将测试集的信息“泄漏”到模型中。
# 没有使用 Scikit-Learn 提供的函数，而是自己手工将数据分割为训练集、验证集、测试集
def three_fold_grid_search():
    X_train_valid, X_test, y_train_valid, y_test = load_train_test_iris()
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, random_state=seed)

    print("训练+验证集合的大小：{}".format(X_train_valid.shape[0]))
    print("训练集合的大小：{}".format(X_train.shape[0]))
    print("验证集合的大小：{}".format(X_valid.shape[0]))
    print("测试集合的大小：{}".format(X_test.shape[0]))

    best_score = 0
    best_parameters = {'C': 0, 'gamma': 0}
    show_title("将数据分割为训练集、验证集、测试集再利用网格搜索对 SVC 模型进行评分")
    from sklearn.svm import SVC
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
            svm = SVC(gamma=gamma, C=C)
            svm.fit(X_train_valid, y_train_valid)
            score = svm.score(X_valid, y_valid)
            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}
                show_subtitle("现在为止的最优参数")
                print("best_parameters =", best_parameters)
                print("best_score =", best_score)

    show_subtitle("搜索到的最佳模型")
    print("最佳参数：{}".format(best_parameters))
    print("验证集上的最佳得分：{:.5f}".format(best_score))
    # 在训练+验证集上重新构建一个模型，并在测试集上进行评估
    svm = SVC(**best_parameters)
    svm.fit(X_train_valid, y_train_valid)
    print("测试集上的最佳得分：{:.5f}".format(svm.score(X_test, y_test)))
    pass


# 5.2.3 带交叉验证的网格搜索
# 利用交叉验证实现训练集和验证集的分割
def cross_validation_grid_search():
    X_train, X_test, y_train, y_test = load_train_test_iris()
    print("训练集合的大小：{}".format(X_train.shape[0]))
    print("测试集合的大小：{}".format(X_test.shape[0]))

    show_title("带交叉验证的网格搜索对 SVC 模型进行评分")
    best_score = 0
    best_parameters = {'C': 0, 'gamma': 0}
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
            svm = SVC(gamma=gamma, C=C)
            # 执行交叉验证
            scores = cross_val_score(svm, X_train, y_train, cv=5)
            # 计算交叉验证的平均精度
            # score = scores.mean 不能使用这个方法，因为返回的是np.array，还需要转换成整数
            score = np.mean(scores)
            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}
                show_subtitle("现在为止的最优参数")
                print("best_parameters =", best_parameters)
                print("best_score =", best_score)

    show_subtitle("搜索到的最佳模型")
    print("最佳参数：{}".format(best_parameters))
    print("验证集上的最佳得分：{:.5f}".format(best_score))
    # 在训练+验证集上重新构建一个模型，并在测试集上进行评估
    svm = SVC(**best_parameters)
    svm.fit(X_train, y_train)
    print("测试集上的最佳得分：{:.5f}".format(svm.score(X_test, y_test)))
    pass


def plot_cross_validation_selection():
    # mglearn.plots.plot_cross_val_selection()
    X_train_val, X_test, y_train_val, y_test = load_train_test_iris()

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
    grid_search.fit(X_train_val, y_train_val)
    results = pd.DataFrame(grid_search.cv_results_)

    best = np.argmax(results.mean_test_score.values)
    plt.figure(figsize=(19, 10))
    # plt.axes((0.12,0.20,0.90,0.88))
    plt.subplots_adjust(bottom=0.20)
    plt.xlim(-1, len(results))
    plt.ylim(0, 1.1)
    marker_cv, marker_mean, marker_best = None, None, None
    for i, (_, row) in enumerate(results.iterrows()):
        scores = row[['split%d_test_score' % i for i in range(5)]]
        marker_cv, = plt.plot([i] * 5, scores, '^', c='gray', markersize=5, alpha=.5)
        marker_mean, = plt.plot(i, row.mean_test_score, 'v', c='none', alpha=1,
                                markersize=10, markeredgecolor='k')
        if i == best:
            marker_best, = plt.plot(i, row.mean_test_score, 'o', c='red',
                                    fillstyle="none", alpha=1, markersize=20,
                                    markeredgewidth=3)

    plt.xticks(range(len(results)),
               [str(x).strip("{}").replace("'", "") for x in grid_search.cv_results_['params']],
               rotation=90)
    plt.ylabel("Validation accuracy")
    plt.xlabel("Parameter settings")
    plt.legend([marker_cv, marker_mean, marker_best],
               ["cv accuracy", "mean accuracy", "best parameter setting"],
               loc='best')
    plt.suptitle("图5-6：带交叉验证的网格搜索结果")


def plot_grid_search_overview():
    # mglearn.plots.plot_grid_search_overview()
    plt.figure(figsize=(10, 8), dpi=70)
    axes = plt.gca()
    axes.yaxis.set_visible(False)
    axes.xaxis.set_visible(False)
    axes.set_frame_on(False)

    def draw(ax, text, start, target=None):
        if target is not None:
            patchB = target.get_bbox_patch()
            end = target.get_position()
        else:
            end = start
            patchB = None
        annotation = ax.annotate(text, end, start, xycoords='axes pixels',
                                 textcoords='axes pixels', size=20,
                                 arrowprops=dict(
                                     arrowstyle="-|>", fc="w", ec="k",
                                     patchB=patchB,
                                     connectionstyle="arc3,rad=0.0"),
                                 bbox=dict(boxstyle="round", fc="w"),
                                 horizontalalignment="center",
                                 verticalalignment="center")
        plt.draw()
        return annotation

    step = 100
    grr = 400

    final_evaluation = draw(axes, "final evaluation", (5 * step, grr - 3 * step))
    retrained_model = draw(axes, "retrained model", (3 * step, grr - 3 * step), final_evaluation)
    best_parameters = draw(axes, "best parameters", (.5 * step, grr - 3 * step), retrained_model)
    cross_validation = draw(axes, "cross-validation", (.5 * step, grr - 2 * step), best_parameters)
    draw(axes, "parameter grid", (0.0, grr - 0), cross_validation)
    training_data = draw(axes, "training data", (2 * step, grr - step), cross_validation)
    draw(axes, "training data", (2 * step, grr - step), retrained_model)
    test_data = draw(axes, "test data", (5 * step, grr - step), final_evaluation)
    draw(axes, "data set", (3.5 * step, grr - 0.0), training_data)
    draw(axes, "data set", (3.5 * step, grr - 0.0), test_data)
    plt.ylim(0, 1)
    plt.xlim(0, 1.5)
    plt.suptitle("图5-7：用GridSearchCV进行参数选择与模型评估的过程概述")


# GridSearchCV类：以估计器的形式实现了这个方法。参数包括（模型，网格参数，交叉验证策略）
# 例子：｛模型：SVC()、网格参数：param_grid、交叉验证策略：cv=5（默认的5折分层交叉验证）
# 为函数提供参数字典，函数会执行所有必要的模型拟合。
# 字典的键：调节的参数名称
# 字典的值：需要尝试的参数设置

def grid_search_cv():
    X_train, X_test, y_train, y_test = load_train_test_iris()

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    print("Parameter grid:{}".format(param_grid))

    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    grid_search = GridSearchCV(SVC(), param_grid, iid=True, cv=5)
    grid_search.fit(X_train, y_train)

    show_title("使用 Scikit-Learn 中 GridSearchCV() 对 SVC 模型进行评分")
    show_subtitle("搜索到的最佳模型")
    print("最佳参数：{}".format(grid_search.best_params_))
    print("最佳估计器：{}".format(grid_search.best_estimator_))
    show_subtitle("模型精度")
    print("在训练集上交叉验证得到的验证集上的最佳得分：{:.5f}".format(grid_search.best_score_))
    print("使用整个训练集训练的模型在测试集上的最佳得分：{:.5f}".format(grid_search.score(X_test, y_test)))
    print("使用整个训练集训练的最佳参数在测试集上的最佳得分：{:.5f}".format(grid_search.best_estimator_.score(X_test, y_test)))

    # 1. 分析交叉验证的结果
    # 转换为DataFrame
    results = pd.DataFrame(grid_search.cv_results_)
    show_subtitle("网格搜索的结果：")
    print(results)

    scores = np.array(results.mean_test_score).reshape(6, 6)

    # SVC对参数设置非常敏感，许多参数精度都在40%以下。
    scores_image = mglearn.tools.heatmap(
        scores, cmap='viridis',
        xlabel='gamma', xticklabels=param_grid['gamma'],
        ylabel='C', yticklabels=param_grid['C'])
    plt.colorbar(scores_image)
    plt.suptitle("图5-8：以C和gamma为自变量，交叉验证平均分数的热图")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    param_grid_linear = {'C': np.linspace(1, 2, 6), 'gamma': np.linspace(1, 2, 6)}
    param_grid_one_log = {'C': np.linspace(1, 2, 6), 'gamma': np.logspace(-3, 2, 6)}
    param_grid_range = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-7, -2, 6)}

    param_grid_list = [param_grid_linear, param_grid_one_log, param_grid_range]
    for param_grid, ax in zip(param_grid_list, axes):
        grid_search = GridSearchCV(SVC(), param_grid, iid=True, cv=5)
        grid_search.fit(X_train, y_train)
        scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)

        scores_image = mglearn.tools.heatmap(
            scores, cmap='viridis', ax=ax,
            xlabel='gamma', xticklabels=param_grid['gamma'],
            ylabel='C', yticklabels=param_grid['C'])
        pass
    plt.colorbar(scores_image, ax=axes.tolist())
    plt.suptitle("图5-9：错误的搜索网格的热图可视化")
    # 第一张图没有显示任何变化，因为参数C和gamma的不正确的缩放和不正确的范围造成的。
    # 第二张图显示的垂直条形模式。似乎只有gamma的设置对精度有影响，而C参数并不重要。
    # 第三张图中C和gamma对应的精度都有变化。而最佳参数出现在右上角，可能搜索范围之外还有更好的参数。


# 2. 在非网格的空间中搜索
# GridSearchCV()支持条件参数，采用的方法依然是字典列表
def non_grid_search_cv():
    X_train, X_test, y_train, y_test = load_train_test_iris()

    show_title("网格列表")
    param_grid_rbf = {'kernel': ['rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    param_grid_linear = {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    param_grid_list = [param_grid_rbf, param_grid_linear]
    print(param_grid_list)

    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    grid_search = GridSearchCV(SVC(), param_grid_list, iid=True, cv=5)
    grid_search.fit(X_train, y_train)

    show_title("使用 GridSearchCV() 的参数列表对 SVC 模型进行评分")
    print("最佳参数：{}".format(grid_search.best_params_))
    print("最佳估计器：{}".format(grid_search.best_estimator_))
    show_subtitle("模型精度")
    print("验证集上的最佳得分：{:.5f}".format(grid_search.best_score_))
    print("测试集上的最佳得分：{:.5f}".format(grid_search.score(X_test, y_test)))

    # 转换为DataFrame
    results = pd.DataFrame(grid_search.cv_results_)
    show_subtitle("网格搜索的结果")
    print(results)
    # 观察结果，会发现linear也有评分第一的参数
    # print(results.params.head())
    # print(results.param_kernel.head())
    # print(results.mean_test_score.head())


# 3. 使用不同的交叉验证策略进行网格搜索
# 交叉验证+分离器+网格搜索+模型+参数列表
# （1）嵌套交叉验证
def grid_search_with_different_strategies():
    from sklearn.datasets import load_iris
    iris = load_iris()

    param_grid_rbf = {'kernel': ['rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    param_grid_linear = {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    param_grid_list = [param_grid_rbf, param_grid_linear]
    show_title("网格列表")
    print(param_grid_list)

    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RepeatedStratifiedKFold
    # 交叉验证的策略
    # 1.分层交叉验证（StratifiedKFold）
    # 2.分层打乱划分交叉验证（StratifiedShuffleSplit）
    # 3.重复分层K折交叉验证分离器（RepeatedStratifiedKFold），进一步增加数据分布的随机性

    from sklearn.model_selection import cross_val_score
    # scores = cross_val_score(grid_search, iris.data, iris.target, cv = 5)
    show_title("使用 GridSearchCV() 的参数列表+交叉验证对 SVC 模型进行评分")
    for cv in [StratifiedKFold(5), StratifiedShuffleSplit(5), RepeatedStratifiedKFold(5)]:
        show_subtitle(str(cv))
        grid_search = GridSearchCV(SVC(), param_grid_list, cv=cv)
        scores = cross_val_score(grid_search, iris.data, iris.target, cv=cv, n_jobs=-1)
        print("SVC在iris数据集上的交叉验证的精度：", scores)
        # [0.967 1.    0.9   0.967 1.   ]->0.9666666666666668
        print("SVC在iris数据集上的交叉验证的平均精度：", scores.mean())


# 自定义的网格搜索算法
def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores = []
    # 外层交叉验证的数据划分训练集和测试集
    for train_samples, test_samples in outer_cv.split(X, y):
        best_params = {}
        best_score = -np.inf
        # 利用内层交叉验证找到最佳参数
        for parameters in parameter_grid:
            # 在内层划分中累加分数
            cv_scores = []
            # 将数据划分训练集和验证集
            # 遍历内层交叉验证
            for inner_train, inner_test in inner_cv.split(X[train_samples], y[train_samples]):
                # 使用给定参数构建分类器
                clf = Classifier(**parameters)
                # 使用给定数据（训练样本中分割出的训练数据）训练分类器
                clf.fit(X[inner_train], y[inner_train])
                # 使用给定数据（训练样本中分割出的测试数据）评估分类器
                score = clf.score(X[inner_test], y[inner_test])
                # 将结果记入得分列表
                cv_scores.append(score)
                pass
            # 计算内层交叉验证的平均分数
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                # 判断是否为最佳分数
                best_score = mean_score
                best_params = parameters
                pass
            pass
        # 使用最佳参数构建分类器
        clf = Classifier(**best_params)
        # 使用给定的数据（训练样本）训练分类器
        clf.fit(X[train_samples], y[train_samples])
        # 使用给定的数据（测试样本）评估分类器
        outer_scores.append(clf.score(X[test_samples], y[test_samples]))
        pass
    return np.array(outer_scores)


def test_nested_cv():
    from sklearn.datasets import load_iris
    iris = load_iris()

    param_grid_rbf = {'kernel': ['rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    param_grid_linear = {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    param_grid_list = [param_grid_rbf, param_grid_linear]
    show_title("网格列表")
    print(param_grid_list)

    from sklearn.model_selection import ParameterGrid, StratifiedKFold
    from sklearn.svm import SVC
    scores = nested_cv(iris.data, iris.target, StratifiedKFold(5), StratifiedKFold(5),
                       SVC, ParameterGrid(param_grid=param_grid_list))
    print("SVC在iris数据集上的交叉验证的精度：", scores)
    #  [0.967 1.    0.967 0.967 1.   ]->0.9800000000000001
    print("SVC在iris数据集上的交叉验证的平均精度：", scores.mean())

    pass


def main():
    # 5.2.1 简单的网格搜索
    # 没有使用 Scikit-Learn 提供的函数，而是自己手工实现的
    # simple_grid_search()
    # 图5 - 5：对数据进行3折划分
    # three_fold_split_data_distribution()
    # 没有使用 Scikit-Learn 提供的函数，而是自己手工将数据分割为训练集、验证集、测试集
    # 手工实现网格搜索
    # three_fold_grid_search()
    # 5.2.3 带交叉验证的网格搜索
    # 利用交叉验证实现训练集和验证集的分割
    # cross_validation_grid_search()
    # 图5-6：带交叉验证的网格搜索结果
    # plot_cross_validation_selection()
    # 图5-7：用GridSearchCV进行参数选择与模型评估的过程概述
    # plot_grid_search_overview()
    # 使用 Scikit-Learn 中 GridSearchCV() 对 SVC 模型进行评分
    # grid_search_cv()
    # 2. 在非网格的空间中搜索
    # non_grid_search_cv()
    # 3. 使用不同的交叉验证策略进行网格搜索
    # grid_search_with_different_strategies()
    # 自定义的网格搜索算法
    # test_nested_cv()
    pass


if __name__ == "__main__":
    main()
    beep_end()
    show_figures()
