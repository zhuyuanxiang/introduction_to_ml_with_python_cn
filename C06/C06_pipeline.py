# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C06_pipeline.py
@Version    :   v0.1
@Time       :   2019-10-12 11:17
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Ch06, P236
@Desc       :   算法链与管道
"""
from datasets.load_data import load_train_test_boston
from datasets.load_data import load_train_test_breast_cancer
from tools import *


# 使用预处理数据训练 SVM 模型
def svm_with_preprocessing_data():
    """训练集中的数据需要进行预处理"""
    X_train, X_test, y_train, y_test = load_train_test_breast_cancer()

    # 对 训练数据集 进行 0-1 缩放 实现 数据预处理
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 在缩放前的 训练数据集 上学习 SVM
    from sklearn.svm import SVC
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)
    show_title("在缩放前的 测试数据集 上评估 SVM")
    print("Test score: {:.2f}".format(svm.score(X_test, y_test)))

    # 在缩放后的 训练数据集 上学习 SVM
    from sklearn.svm import SVC
    svm = SVC(gamma='auto')
    svm.fit(X_train_scaled, y_train)
    show_title("在缩放后的 测试数据集 上评估 SVM")
    print("Test score: {:.2f}".format(svm.score(X_test_scaled, y_test)))


# 6.1 用预处理进行参数选择
def parameter_selection():
    # 加载然后划分数据
    X_train, X_test, y_train, y_test = load_train_test_breast_cancer()
    show_title("不使用管道进行参数选择")

    # 对训练数据集进行缩放
    # 使用全部训练数据集的信息进行缩放学习，
    # 那么后面的交叉验证过程中验证集中数据已经把部分信息“泄漏”给了交叉验证中训练集中的数据。
    # 因此缩放必须在先划分出验证集后再进行训练（即在进行任何预处理之前完成数据集的划分）
    # 后面使用Pipeline就可以实现这个功能，而当前代码中无法正确实现。
    from sklearn.preprocessing import MinMaxScaler
    # 使用0-1缩放进行预处理的数据
    scaler = MinMaxScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 在缩放后的训练数据集上学习SVM
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train)
    print("验证集上的最佳精度 (Best cross-validation accuracy): {:.2f}".format(grid_search.best_score_))
    print("验证集上的最佳参数 (Best parameters): ", grid_search.best_params_)
    print("验证集上的最佳估计器 (SVM estimator): ", grid_search.best_estimator_)
    print("测试集上的精度 (Best test set score): {:.2f}".format(grid_search.score(X_test_scaled, y_test)))


def plot_data_preprocess():
    # 交叉验证与最终评估这两个过程中数据处理存在的不同
    mglearn.plots.plot_improper_processing()
    plt.suptitle("图6-1：在交叉验证循环外部对数据进行预处理时的数据使用情况\n"
                 "因为显示尺度问题，建议看书上的图更准确")


# 6.2 构建管道
# Pipeline类可以将多个处理步骤合并（glue）为单个Scikit-Learn的估计器。
# Pipeline类也提供了fit()、predict()和score()函数，其行为与其他模型相同。
# Pipeline最常见的用例是将预处理步骤（比如：数据缩放）与一个监督模型（比如：分类器）链接在一起
def construct_pipeline():
    # 加载然后划分数据
    X_train, X_test, y_train, y_test = load_train_test_breast_cancer()
    show_title("使用管道基于固定参数实现数据预处理和模型训练")

    # 对训练数据集进行缩放
    # 使用全部训练数据集的信息进行缩放学习，
    # 那么后面的交叉验证过程中验证集中数据已经把部分信息“泄漏”给了交叉验证中训练集中的数据。
    # 因此缩放必须在先划分出验证集后再进行训练（即在进行任何预处理之前完成数据集的划分）
    # 使用Pipeline就可以实现这个功能（结合 parameter_selection()理解Pipeline）
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC

    from sklearn.pipeline import Pipeline
    # 创建了两个步骤：数据预处理（0-1缩放）和监督模型（SVM）
    # Current default is 'auto' which uses 1 / n_features
    pipe_line = Pipeline([('scaler', MinMaxScaler()), ("svm", SVC(C=1.0, gamma='auto'))])  # SVC(gamma = 1, C = 1))])
    pipe_line.fit(X_train, y_train)
    print("Test score: {:.2f}".format(pipe_line.score(X_test, y_test)))

    svm = pipe_line.named_steps['svm']
    print("SVM parameters: (C:{}, gamma: {})".format(svm.C, svm.gamma))
    print("SVM Model: ", svm)


# 6.3 在网格搜索中使用管道
def pipeline_in_grid_search():
    # 加载然后划分数据
    X_train, X_test, y_train, y_test = load_train_test_breast_cancer()
    show_title("使用管道进行参数选择")

    # 对训练数据集进行缩放
    # 使用全部训练数据集的信息进行缩放学习，
    # 那么后面的交叉验证过程中验证集中数据已经把部分信息“泄漏”给了交叉验证中训练集中的数据。
    # 因此缩放必须在先划分出验证集后再进行训练（即在进行任何预处理之前完成数据集的划分）
    # 使用Pipeline就可以实现这个功能（结合 parameter_selection()理解Pipeline）

    # 使用0-1缩放进行预处理的数据
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    # 创建了两个步骤：数据预处理（0-1缩放）和监督模型（SVM）
    pipe_line = Pipeline([('scaler', MinMaxScaler()), ("svm", SVC(gamma='auto'))])
    pipe_line.fit(X_train, y_train)

    # 在网格搜索中使用管道
    from sklearn.model_selection import GridSearchCV
    param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(pipe_line, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("验证集上的最佳精度 (Best cross-validation accuracy): {:.2f}".format(grid_search.best_score_))
    print("验证集上的最佳参数 (Best parameters): ", grid_search.best_params_)
    print("验证集上的最佳估计器 (SVM estimator): ", grid_search.best_estimator_)
    print("测试集上的精度 (Best test set score): {:.2f}".format(grid_search.score(X_test, y_test)))


def plot_inner_preprocess():
    # 输出的结果与图6-1处理的结果没有区别
    # 因为在交叉验证中，信息泄漏的影响取决于预处理步骤的性质。
    # 如果是使用测试部分来估计数据的范围，一般的影响不大；
    # 如果是使用测试部分来提取特征或者选择特征，那么影响较大。
    mglearn.plots.plot_proper_processing()
    plt.suptitle("图6-2：在交叉验证循环内部使用管道对数据进行预处理时的数据使用情况")


# 信息泄露的例子
def information_leak():
    # 这是个回归任务，数据（高斯分布中独立采样的100个样本与10000个特征，高斯分布中采样的100个响应）
    rnd = np.random.RandomState(seed=0)
    X = rnd.normal(size=(100, 10000))
    y = rnd.normal(size=(100,))

    number_title = "信息泄露"
    print('\n', '-' * 5, number_title, '-' * 5)

    # 基于一元线性回归检测从10000个特征中选择500个特征（5%）
    from sklearn.feature_selection import SelectPercentile, f_regression

    select = SelectPercentile(score_func=f_regression, percentile=5)
    # select = SelectPercentile(percentile = 5)
    # 使用所有特征进行训练，从中选择500个特征
    select.fit(X, y)
    X_selected = select.transform(X)
    print("X_selected.shape: {}".format(X_selected.shape))

    from sklearn.model_selection import cross_val_score
    # 基于一元线性回归检测取出的特征具有较强的线性相关性，
    # 这样的特征基于Ridge()线性模型来学习，就可以得到较高的精确度
    # 这样的特征基于SVR()模型来学习，就无法得到很高的精确度，但是依然比实际情况高许多
    from sklearn.linear_model import Ridge
    print('-' * 20)
    print("Cross-validation accuracy(cv only on ridge): {:.2f}".format(
            np.mean(cross_val_score(Ridge(),
                                    X_selected, y, cv=5))
    ))
    from sklearn.svm import SVR
    print("Cross-validation accuracy(cv only on svm): {:.2f}".format(
            np.mean(cross_val_score(SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.2),
                                    X_selected, y, cv=5))
    ))

    from sklearn.pipeline import Pipeline
    print('-' * 20)
    # 创建了两个步骤：数据预处理（0-1缩放）和监督模型（Ridge）
    # 使用管道才会得到准确的交叉验证的结果，即模型效果很差
    # 因为仅使用数据的训练部分来选择特征进行训练，那么模型对于测试部分进行评估时就会发现效果很差
    pipe_line = Pipeline([('select', SelectPercentile(score_func=f_regression, percentile=5)),
                          ("ridge", Ridge())])
    print("Cross-validation accuracy (pipeline on ridge): {:.2f}".format(
            np.mean(cross_val_score(pipe_line, X, y, cv=5))
    ))
    # 创建了两个步骤：数据预处理（0-1缩放）和监督模型（SVR）
    pipe_line = Pipeline([('select', SelectPercentile(score_func=f_regression, percentile=5)),
                          ("svm", SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.2))])
    print("Cross-validation accuracy (pipeline on svm): {:.2f}".format(
            np.mean(cross_val_score(pipe_line, X, y, cv=5))
    ))
    pass


# 6.4 通用的管道接口
def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:-1]:
        # 遍历除最后一步之外的所有步骤对数据进行拟合和变换
        X_transformed = estimator.fit_transform(X_transformed, y)
    # 对最后一步进行拟合
    self.steps[-1][1].fit(X_transformed, y)
    return self


def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        # 遍历除最后一步之外的所有步骤对数据进行变换
        X_transformed = step[1].transform(X_transformed)
    # 利用最后一步进行预测
    return self.steps[-1][1].predict(X_transformed)


# 6.4.1 使用make_pipeline创建管道
# 6.4.2 访问管道中某个步骤的属性
def create_pipeline_methods():
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC

    show_title("创建管道")

    # 创建Pipeline的标准语法
    from sklearn.pipeline import Pipeline
    pipeline_long = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC(C=100))])
    print("Long pipeline steps: {}".format(pipeline_long.steps))

    # 创建Pipeline的缩写语法
    from sklearn.pipeline import make_pipeline
    pipeline_short = make_pipeline(MinMaxScaler(), SVC(C=100))
    print("Short pipeline steps: {}".format(pipeline_short.steps))

    # 创建属于同一个类的多个步骤
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    show_subtitle("创建属于同一个类的多个步骤的管道")

    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
    print("Pipeline steps(same class): {}".format(pipeline.steps))

    # 建议使用具有明确名称的Pipeline构建，方便给每个步骤提供更具语义的名称。

    X_train, X_test, y_train, y_test = load_train_test_breast_cancer()
    pipeline.fit(X_train)
    components = pipeline.named_steps['pca'].components_
    print("pipeline.named_steps['pca'] = ", pipeline.named_steps['pca'])
    print("pipeline.named_steps['pca'].components_.shape: {}".format(components.shape))
    print("pipeline.named_steps['pca'].components_:")
    print(components)
    pass


# 6.4.3 访问管道中某个网格搜索中的属性
def pipeline_attributes():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    show_title("访问管道中某个网格搜索中的属性")

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=10000))
    print("Pipeline steps:", pipeline.steps)
    param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}

    X_train, X_test, y_train, y_test = load_train_test_breast_cancer()

    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    show_subtitle("Best estimator")
    print(grid_search.best_estimator_)

    show_subtitle("Best estimator : Logistic regression step")
    print(grid_search.best_estimator_.named_steps['logisticregression'])

    show_subtitle("Best estimator : Logistic regression coefficients")
    print(grid_search.best_estimator_.named_steps['logisticregression'].coef_)

    pass


# 6.5 网格搜索预处理步骤与模型参数
# 利用管道将机器学习工作流程中的所有处理步骤封闭成一个Scikit-Learn估计器，可以使用监督任务的输出来调节预处理参数
def adjust_preprocess_parameter():
    """管道中增加了特征生成（多项式特征）步骤"""
    X_train, X_test, y_train, y_test = load_train_test_boston()

    show_title("数据预处理+多项式特征+训练模型")

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV

    param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    param_grid_poly = {'polynomialfeatures__degree': [1, 2, 3], 'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    pipeline = make_pipeline(StandardScaler(), Ridge())
    pipeline_poly = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=3, iid=True)
    grid_search_poly = GridSearchCV(pipeline_poly, param_grid=param_grid_poly, cv=5, n_jobs=3, iid=True)

    grid_search.fit(X_train, y_train)
    grid_search_poly.fit(X_train, y_train)

    show_subtitle("pipeline without poly features")
    print("Best parameters :", grid_search.best_params_)
    print("Test-set score : {:.2f}".format(grid_search.score(X_test, y_test)))

    show_subtitle("pipeline with poly features")
    print("Best parameters :", grid_search_poly.best_params_)
    print("Test-set score : {:.2f}".format(grid_search_poly.score(X_test, y_test)))

    plt.matshow(grid_search_poly.cv_results_['mean_test_score'].reshape(3, -1), vmin=0, cmap='viridis')
    plt.xlabel('岭回归的alpha值')
    plt.ylabel('多项式特征的度')
    plt.xticks(range(len(param_grid_poly['ridge__alpha'])), param_grid_poly['ridge__alpha'])
    plt.yticks(range(len(param_grid_poly['polynomialfeatures__degree'])), param_grid_poly['polynomialfeatures__degree'])
    plt.colorbar()
    plt.suptitle("图6-4：以多项式特征的次数和岭回归的alpha参数为坐标轴来绘制交叉验证平均分数的热图")


# 6.6 利用网格搜索和管道选择模型
def model_selection_with_grid_pipe():
    X_train, X_test, y_train, y_test = load_train_test_breast_cancer()
    show_title("利用网格搜索和管道选择模型")

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    # 这里初始化为了正确实例化管道。
    # 注1：最后一个估计器必须提供 score() 函数
    # 注2：memory 参数提供缓存模型的路径，避免重复计算
    # pipeline = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC(gamma = 'auto'))])
    pipeline = Pipeline([('preprocessing', None), ('classifier', SVC(gamma='auto'))], memory='cache')
    # pipeline = Pipeline([('preprocessing', None), ('classifier', None)])  # Error
    param_grid = [{
            'classifier': [SVC(gamma='auto')],
            'preprocessing': [StandardScaler(), MinMaxScaler(), None],
            'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
    },
            {
                    'classifier': [RandomForestClassifier(n_estimators=100)],
                    'preprocessing': [None],
                    'classifier__max_features': [1, 2, 3]
            }]

    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=3, iid=True)
    grid.fit(X_train, y_train)

    print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
    print("Best set score: {:.2f}".format(grid.score(X_test, y_test)))
    print("Best parameters: ", grid.best_params_)
    print("Pipeline: ", pipeline)
    # 系统最终选择是SVC模型，而不是随机森林


def main():
    # 使用预处理数据训练 SVM 模型
    # svm_with_preprocessing_data()
    # 6.1 用预处理进行参数选择
    # parameter_selection()
    # plot_data_preprocess()
    # 6.2 构建管道
    # construct_pipeline()
    # 6.3 在网格搜索中使用管道
    # pipeline_in_grid_search()
    # plot_inner_preprocess()
    # 信息泄露的例子
    # information_leak()
    # 6.4 通用的管道接口
    # 6.4.1 使用make_pipeline创建管道
    # 6.4.2 访问管道中某个步骤的属性
    # create_pipeline_methods()
    # 6.4.3 访问管道中某个网格搜索中的属性
    # pipeline_attributes()
    # 6.5 网格搜索预处理步骤与模型参数
    # adjust_preprocess_parameter()
    # 6.6 利用网格搜索和管道选择模型
    model_selection_with_grid_pipe()
    pass


if __name__ == "__main__":
    main()
    beep_end()
    show_figures()
