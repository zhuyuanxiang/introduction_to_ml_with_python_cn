# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C0503_evaluate.py
@Version    :   v0.1
@Time       :   2019-10-12 10:44
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec0503，P213
@Desc       :   模型评估与改进。评估指标与评分
"""
from tools import *


def evaluate_imbalanced_dataset():
    """不平衡的数据使得预测结果无法说明模型预测到底有多少改进"""
    from sklearn.datasets import load_digits
    digits = load_digits()
    # print(np.bincount(digits.target))
    # [178 182 177 183 181 182 181 179 174 180]
    # print(np.unique(digits.target))
    # [0 1 2 3 4 5 6 7 8 9]
    y = (digits.target == 9)

    show_title("不平衡的数据对模型的影响")
    print("数据中90%都是False，只有10%的数据是True，因此只要预测False，90%都是正确的")

    # 将数据划分为训练集和测试集，是为了利用测试集度量模型的泛化能力。
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=seed)
    show_subtitle("测试数据集的标签")
    print(y_test)

    from sklearn.dummy import DummyClassifier
    # DummyClassifier使用最简单的规则来预测
    # strategy = 'most_frequent'，使用数据中频率最高的类别作为预测的目标
    dummy_majority = DummyClassifier(strategy='most_frequent')
    dummy_majority.fit(X_train, y_train)
    predict_most_frequent = dummy_majority.predict(X_test)
    show_subtitle("模型预测结果")
    print("唯一被预测的标签是: {}".format(np.unique(predict_most_frequent)))
    # Unique predicted labels: [False]
    print("测试数据使用频率模型预测的结果: {:.2f}".format(dummy_majority.score(X_test, y_test)))
    # Most frequent test score: 0.90

    # strategy = "stratified"，默认策略，基于训练数据的概率分布
    dummy = DummyClassifier(strategy="stratified")
    dummy.fit(X_train, y_train)
    print("测试数据使用训练数据概率分布模型预测的结果: {:.2f}".format(dummy.score(X_test, y_test)))
    # Dummy test score: 0.84

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X_train, y_train)
    print("测试数据使用决策树模型预测的结果: {:.2f}".format(tree.score(X_test, y_test)))
    # Decision Tree test score: 0.92

    from sklearn.linear_model import LogisticRegression
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=10000, C=0.1)
    logistic_regression.fit(X_train, y_train)
    print("测试数据使用 Logistic Regression 模型预测的结果: {:.2f}".format(logistic_regression.score(X_test, y_test)))
    # Logistic Regression test score: 0.98
    pass


def compare_confusion_matrix():
    from sklearn.datasets import load_digits
    digits = load_digits()
    # print(np.bincount(digits.target))
    # [178 182 177 183 181 182 181 179 174 180]
    # print(np.unique(digits.target))
    # [0 1 2 3 4 5 6 7 8 9]
    y = (digits.target == 9)

    show_title("使用混淆矩阵分析不平衡的数据对模型的影响")
    print("数据中90%都是False，只有10%的数据是True，因此只要预测False，90%都是正确的")

    # 将数据划分为训练集和测试集，是为了利用测试集度量模型的泛化能力。
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=seed)
    print(y_test)

    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import confusion_matrix
    # DummyClassifier使用最简单的规则来预测
    # strategy = 'most_frequent'，使用数据中频率最高的类别作为预测的目标
    dummy_majority = DummyClassifier(strategy='most_frequent')
    dummy_majority.fit(X_train, y_train)
    predict_most_frequent = dummy_majority.predict(X_test)
    print("频率模型的混淆矩阵:")
    print(confusion_matrix(y_test, predict_most_frequent))

    # strategy = "stratified"，默认策略，基于训练数据的概率分布
    dummy = DummyClassifier(strategy="stratified")
    dummy.fit(X_train, y_train)
    predict_dummy = dummy.predict(X_test)
    print("训练数据概率分布模型的混淆矩阵:")
    print(confusion_matrix(y_test, predict_dummy))

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X_train, y_train)
    predict_tree = tree.predict(X_test)
    print("决策树模型的混淆矩阵:")
    print(confusion_matrix(y_test, predict_tree))

    from sklearn.linear_model import LogisticRegression
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=10000, C=0.1)
    logistic_regression.fit(X_train, y_train)
    predict_logistic_regression = logistic_regression.predict(X_test)
    print("Logistic Regression 模型的混淆矩阵:")
    print(confusion_matrix(y_test, predict_logistic_regression))

    print("混淆矩阵显示 Logistic Regression 模型的预测效果最好，因为数据最平衡")


def show_confusion_matrix():
    mglearn.plots.plot_confusion_matrix_illustration()
    plt.figure(figsize=(10, 8))
    confusion = np.array([[401, 2], [8, 39]])
    plt.text(.25, .7, confusion[0, 0], size=50, horizontalalignment='center', verticalalignment="center")
    plt.text(.25, .2, confusion[1, 0], size=50, horizontalalignment='center', verticalalignment="center")
    plt.text(.75, .7, confusion[0, 1], size=50, horizontalalignment='center', verticalalignment="center")
    plt.text(.75, .2, confusion[1, 1], size=50, horizontalalignment='center', verticalalignment="center")
    plt.xticks([.25, .75], ["预测结果\n'非九'", "预测结果\n'是九'"], size=15)
    plt.yticks([.25, .75], ["真实数据\n'是九'", "真实数据\n'非九'"], size=15)
    plt.plot([.5, .5], [0, 1], '--', c='k')
    plt.plot([0, 1], [.5, .5], '--', c='k')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.suptitle("图5-10：“9与其他”分类任务基于 Logistic Regression 模型的混淆矩阵")

    plt.figure()
    # mglearn.plots.plot_binary_confusion_matrix()
    plt.text(.25, .6, "TN", size=100, horizontalalignment='center')
    plt.text(.25, .1, "FN", size=100, horizontalalignment='center')
    plt.text(.75, .6, "FP", size=100, horizontalalignment='center')
    plt.text(.75, .1, "TP", size=100, horizontalalignment='center')
    plt.xticks([.25, .75], ["正预测", "负预测"], size=15)
    plt.yticks([.25, .75], ["正类", "负类"], size=15)
    plt.plot([.5, .5], [0, 1], '--', c='k')
    plt.plot([0, 1], [.5, .5], '--', c='k')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.suptitle("图5-11：二分类混淆矩阵")


def compare_f1_score():
    from sklearn.datasets import load_digits

    digits = load_digits()
    # print(np.bincount(digits.target))
    # [178 182 177 183 181 182 181 179 174 180]
    # print(np.unique(digits.target))
    # [0 1 2 3 4 5 6 7 8 9]
    y = (digits.target == 9)

    show_title("使用F1分数分析不平衡的数据对模型的影响")
    print("数据中90%都是False，只有10%的数据是True，因此只要预测False，90%都是正确的")

    # 将数据划分为训练集和测试集，是为了利用测试集度量模型的泛化能力。
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=seed)
    show_subtitle("测试集的标签")
    print(y_test)

    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    # DummyClassifier使用最简单的规则来预测
    # strategy = 'most_frequent'，使用数据中频率最高的类别作为预测的目标
    dummy_majority = DummyClassifier(strategy='most_frequent')
    dummy_majority.fit(X_train, y_train)
    predict_most_frequent = dummy_majority.predict(X_test)
    print("频率模型的F1分数: {:.2f}".format(f1_score(y_test, predict_most_frequent)))
    # f1 score most frequent: 0.00

    # strategy = "stratified"，默认策略，基于训练数据的概率分布
    dummy = DummyClassifier(strategy="stratified")
    dummy.fit(X_train, y_train)
    predict_dummy = dummy.predict(X_test)
    print("训练数据概率分布模型的F1分数: {:.2f}".format(f1_score(y_test, predict_dummy)))
    # f1 score dummy: 0.12

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X_train, y_train)
    predict_tree = tree.predict(X_test)
    print("决策树模型的F1分数: {:.2f}".format(f1_score(y_test, predict_tree)))
    # f1 score decision tree: 0.55

    from sklearn.linear_model import LogisticRegression
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=10000, C=0.1)
    logistic_regression.fit(X_train, y_train)
    predict_logistic_regression = logistic_regression.predict(X_test)
    print("Logistic Regression 模型的F1分数: {:.2f}".format(f1_score(y_test, predict_logistic_regression)))
    # f1 score logistic regression: 0.92
    pass


# 使用分类报告分析不平衡的数据对模型的影响
def compare_classification_report():
    from sklearn.datasets import load_digits

    digits = load_digits()
    # print(np.bincount(digits.target))
    # [178 182 177 183 181 182 181 179 174 180]
    # print(np.unique(digits.target))
    # [0 1 2 3 4 5 6 7 8 9]
    y = (digits.target == 9)
    target_names = ["not nine", 'nine']
    # target_names = ["非九", '是九']   // 中文会影响显示的对齐

    show_title("使用分类报告分析不平衡的数据对模型的影响")
    print("分类报告（准确率、召回率、F1分数、支持的数据个数")
    print("数据中90%都是False，只有10%的数据是True，因此只要预测False，90%都是正确的")

    # 将数据划分为训练集和测试集，是为了利用测试集度量模型的泛化能力。
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=seed)
    show_subtitle("测试集的标签")
    print(y_test)

    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import classification_report
    # DummyClassifier使用最简单的规则来预测
    # strategy = 'most_frequent'，使用数据中频率最高的类别作为预测的目标
    dummy_majority = DummyClassifier(strategy='most_frequent')
    dummy_majority.fit(X_train, y_train)
    predict_most_frequent = dummy_majority.predict(X_test)
    show_subtitle("频率模型的分类报告")
    print(classification_report(y_test, predict_most_frequent, target_names=target_names))

    # strategy = "stratified"，默认策略，基于训练数据的概率分布
    dummy = DummyClassifier(strategy="stratified")
    dummy.fit(X_train, y_train)
    predict_dummy = dummy.predict(X_test)
    show_subtitle("训练数据概率分布模型的分类报告")
    print(classification_report(y_test, predict_dummy, target_names=target_names))
    # ToDo: 为什么分类报告给出的F1结果与直接f1_score()的结果不一样？

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X_train, y_train)
    predict_tree = tree.predict(X_test)
    show_subtitle("决策树模型的分类报告")
    print(classification_report(y_test, predict_tree, target_names=target_names))

    from sklearn.linear_model import LogisticRegression
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=10000, C=0.1)
    logistic_regression.fit(X_train, y_train)
    predict_logistic_regression = logistic_regression.predict(X_test)
    show_subtitle("Logistic Regression 模型的分类报告")
    print(classification_report(y_test, predict_logistic_regression, target_names=target_names))

    pass


# 不平衡数据的二分类问题
def imbalanced_two_classes():
    from mglearn.datasets import make_blobs
    # X, y = make_blobs(n_samples=(350, 50), centers=[2], cluster_std=[7.0, 2], random_state=seed)
    X, y = make_blobs(n_samples=(350, 50), cluster_std=[7.0, 2], random_state=seed)

    show_title("不平衡数据的二分类问题")
    print("数据中87.5%是一类，12.5%的数据是另一类")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    show_subtitle("测试集的标签")
    print(y_test)

    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    svc = SVC(gamma=0.05)
    svc.fit(X_train, y_train)
    predict_svc = svc.predict(X_test)
    show_subtitle("SVC分类报告")
    print(classification_report(y_test, predict_svc))

    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_test, predict_svc)
    show_subtitle("SVC分类的混淆矩阵")
    print(confusion)

    # 使用决策函数可以调整数据的平衡问题，以及样本中不同类别的权重
    # 不过这种人工设置阈值的方式不是很好
    predict_svc_lower_threshold = svc.decision_function(X_test) > -0.35
    show_subtitle("SVC基于决策函数进行预测的分类报告：")
    print(classification_report(y_test, predict_svc_lower_threshold))
    # 决策函数的阈值选取，只能依靠经验，没有合适的算法

    confusion = confusion_matrix(y_test, predict_svc_lower_threshold)
    show_subtitle("SVC分类的混淆矩阵")
    print(confusion)

    mglearn.plots.plot_decision_threshold()
    plt.suptitle("图5-12：决策函数的热图与改变决策阈值的影响")
    pass


# 5. 准确率——召回率曲线
def compare_precision_recall_curve():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=(4000, 500), n_features=2, cluster_std=[7.0, 2], random_state=22)

    show_title("使用“准确率——召回率曲线”分析不平衡的数据对模型的影响")
    print("数据中87.5%是一类，12.5%的数据是另一类")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    show_subtitle("测试集的标签")
    print(y_test)

    from sklearn.svm import SVC
    svc = SVC(gamma=0.05)
    svc.fit(X_train, y_train)

    from sklearn.metrics import precision_recall_curve
    precision_svc, recall_svc, thresholds_svc = precision_recall_curve(y_test, svc.decision_function(X_test))

    # 找到最接近于0的阈值的位置
    close_zero = np.argmin(np.abs(thresholds_svc))
    plt.plot(precision_svc[close_zero], recall_svc[close_zero], 'o',
             markersize=10, label="SVC的0阈值",
             fillstyle='none', c='k', mew=2)
    plt.plot(precision_svc, recall_svc, label="准确率——召回率曲线")
    plt.xlabel('准确率')
    plt.ylabel('召回率')
    plt.legend()
    plt.suptitle("图5-13：SVC（gamma=0.05）的准确率--召回率曲线\n"
                 "曲线越靠近右上角，则分类器越好")

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
    rf.fit(X_train, y_train)

    # RandomForestClassifier 有预测概率（predict_proba），但是没有决策函数（decision_function）
    precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])

    plt.figure()
    plt.plot(precision_svc, recall_svc, label="SVC")
    plt.plot(precision_svc[close_zero], recall_svc[close_zero], 'o',
             markersize=10, label="SVC的0阈值",
             fillstyle='none', c='k', mew=2)

    close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
    plt.plot(precision_rf, recall_rf, label="随机森林")
    plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^',
             markersize=10, label="随机森林的0.5阈值",
             fillstyle='none', mew=2)
    plt.xlabel('准确率')
    plt.ylabel('召回率')
    plt.legend()
    plt.title("图5-14：比较 SVM 与 随机森林 的 准确率--召回率曲线\n"
              "SVM在中间位置的表现更好\n"
              "随机森林在极值处表现更好（即极值处的精度或是高准确率或是高召回率）")

    show_subtitle("f1_score表示了准确率——召回率曲线上默认阈值对应的点")
    from sklearn.metrics import f1_score
    predict_svc = svc.predict(X_test)
    print("SVC的f1_score: {:.3f}".format(f1_score(y_test, predict_svc)))
    predict_rf = rf.predict(X_test)
    print("随机森林的f1_score: {:.3f}".format(f1_score(y_test, predict_rf)))

    show_subtitle("平均准确率（Average Precision）表示曲线下的积分（即面积）")
    from sklearn.metrics import average_precision_score
    ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
    print("SVC的平均准确率：{:.3f}".format(ap_svc))
    ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
    print("随机森林的平均准确率：{:.3f}".format(ap_rf))

    pass


# 6. 受试者工作特征（ROC）与 AUC
# ROC（Receiver Operating Characteristics Curve）
# AUC（Area Under the Curve）
def compare_roc_curve():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=(4000, 500), n_features=2, cluster_std=[7.0, 2], random_state=22)

    show_title("使用 ROC 曲线分析不平衡的数据对模型的影响")
    print("数据中87.5%是一类，12.5%的数据是另一类")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    show_subtitle("测试集的标签")
    print(y_test)

    from sklearn.svm import SVC
    svc = SVC(gamma=0.05)
    svc.fit(X_train, y_train)

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))

    # 找到最接近于0的阈值的位置
    close_zero = np.argmin(np.abs(thresholds))

    plt.plot(fpr[close_zero], tpr[close_zero], 'o',
             markersize=10, label="SVC的0阈值",
             fillstyle='none', c='k', mew=2)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel('FPR')
    plt.ylabel('TPR(recall)')
    plt.legend()
    plt.title("图5-15：SVC（gamma=0.05）的ROC曲线\n"
              "曲线越靠近左上角，则分类器越好")

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
    rf.fit(X_train, y_train)

    # RandomForestClassifier有predict_proba，但是没有decision_function
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
    close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))

    plt.figure()
    plt.plot(fpr[close_zero], tpr[close_zero], 'o',
             markersize=10, label="SVC的0阈值",
             fillstyle='none', c='k', mew=2)
    plt.plot(fpr, tpr, label="ROC Curve SVC")

    plt.plot(fpr_rf[close_default_rf], tpr_rf[close_default_rf], '^',
             markersize=10, label="随机森林的0.5阈值",
             fillstyle='none', c='k', mew=2)
    plt.plot(fpr_rf, tpr_rf, label="ROC Curve RF")

    plt.xlabel('FPR（假真类率）')
    plt.ylabel('TPR（真真类率）')
    plt.legend()
    plt.title("图5-16：比较 SVM 和 随机森林 的 ROC曲线\n"
              "曲线越靠近左上角，则分类器越好\n"
              "即假真类率（FPR）要低，真真类率（TPR）要高")

    # 对于不平衡数据集的分类问题，AUC指标比精度指标的效果更好。
    # 分别随机从样本集中抽取一个正样本和一个负样本，正样本的预测值大于负样本的预测值的概率。
    show_subtitle("AUC 表示曲线下的积分（即面积），解释为评估正例样本的排名")
    from sklearn.metrics import roc_auc_score
    svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
    print("SVC的AUC：{:.3f}".format(svc_auc))
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print("随机森林的AUC：{:.3f}".format(rf_auc))
    print("对于不平衡类别的分类问题，选择模型时使用 AUC 比 精度 更有意义")
    pass


# 对比 SVC 的不同 gamma 值条件下的 ROC 曲线，从而选择最优参数配置
def compare_svc_gamma_roc():
    from sklearn.datasets import load_digits

    digits = load_digits()
    # print(np.bincount(digits.target))
    # [178 182 177 183 181 182 181 179 174 180]
    # print(np.unique(digits.target))
    # [0 1 2 3 4 5 6 7 8 9]
    y = (digits.target == 9)

    show_title("对比 SVC 模型中不同 gamma 值对 ROC 曲线的影响")
    print("数据中90%都是False，只有10%的数据是True，因此只要预测False，90%都是正确的")

    # 将数据划分为训练集和测试集，是为了利用测试集度量模型的泛化能力。
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=seed)

    from sklearn.svm import SVC
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    show_title("所有精确度都是一样的， AUC 是不同的。")
    plt.figure()
    for gamma in [1, 0.5, 0.1, 0.05, 0.01]:
        svc = SVC(gamma=gamma)
        svc.fit(X_train, y_train)
        accuracy = svc.score(X_test, y_test)
        fpr, tpr, _ = roc_curve(y_test, svc.decision_function(X_test))
        auc = roc_auc_score(y_test, svc.decision_function(X_test))
        gamma_title = "gamma = {:.5f}".format(gamma)
        show_subtitle(gamma_title)
        print("精确度 = {:.5f}\t AUC = {:.5f}".format(accuracy, auc))
        plt.plot(fpr, tpr, label=gamma_title)
        pass
    plt.xlabel('FPR（假真类率）')
    plt.ylabel('TPR（真真类率）')
    plt.xlim(-0.01, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="best")
    plt.title("图5-17：对比不同gamma值的SVM的ROC曲线")
    pass


# 5.3.3 多分类指标
def multi_classes_score():
    from sklearn.datasets import load_digits

    # 数字图片
    digits = load_digits()
    # print(np.bincount(digits.target))
    # [178 182 177 183 181 182 181 179 174 180]
    # print(np.unique(digits.target))
    # [0 1 2 3 4 5 6 7 8 9]

    show_title("基于混淆矩阵对多分类指标评估")

    # 将数据划分为训练集和测试集，是为了利用测试集度量模型的泛化能力。
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=seed)

    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=10000, C=0.1, multi_class='auto')
    logistic_regression.fit(X_train, y_train)
    predict_logistic_regression = logistic_regression.predict(X_test)
    print("精确度: {:.3f}".format(accuracy_score(y_test, predict_logistic_regression)))
    show_subtitle("混淆矩阵")
    confusion_matrix_logistic_regression = confusion_matrix(y_test, predict_logistic_regression)
    print(confusion_matrix_logistic_regression)

    scores_image = mglearn.tools.heatmap(
        confusion_matrix_logistic_regression, xlabel="Predicted Label", ylabel="True Label",
        xticklabels=digits.target_names, yticklabels=digits.target_names,
        cmap='gray_r', fmt='%d')
    plt.title("混淆矩阵")
    plt.gca().invert_yaxis()
    plt.title("图5-18：10个数字分类任务的混淆矩阵")

    show_subtitle("10个数字分类任务")
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report
    print("微平均(micro avg)：{:.3f}".format(f1_score(y_test, predict_logistic_regression, average='micro')))
    print("宏平均(macro avg)：{:.3f}".format(f1_score(y_test, predict_logistic_regression, average='macro')))
    print("加权平均(weighted avg)：{:.3f}".format(f1_score(y_test, predict_logistic_regression, average='weighted')))
    show_subtitle("10个数字分类任务的分类报告")
    print(classification_report(y_test, predict_logistic_regression))


# 5.3.5 在模型选择中使用评估指标
def model_selection_with_score():
    from sklearn.datasets import load_digits

    # 数字图片
    digits = load_digits()
    # print(np.bincount(digits.target))
    # [178 182 177 183 181 182 181 179 174 180]
    # print(np.unique(digits.target))
    # [0 1 2 3 4 5 6 7 8 9]

    # 将数据划分为训练集和测试集，是为了利用测试集度量模型的泛化能力。
    from sklearn.model_selection import train_test_split
    y = (digits.target == 9)
    X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=seed)

    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC
    cross_val = cross_val_score(SVC(gamma='auto'), digits.data, y, scoring='accuracy', cv=5)
    show_title("交叉验证的默认评估指标是：accuracy")
    print("Accuracy scoring:", cross_val)
    cross_val = cross_val_score(SVC(gamma='auto'), digits.data, y, scoring='roc_auc', cv=5)
    show_subtitle("交叉验证的默认评估指标是：roc_auc")
    print("AUC scoring: ", cross_val)

    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score
    param_grid = {'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}

    np.set_printoptions(precision=5, suppress=True, threshold=np.inf, linewidth=200)

    # 使用 精确度 评分
    grid_search = GridSearchCV(SVC(gamma='auto'), param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    show_title(f"网格搜索的评估指标：{grid_search.scoring}")
    print("Best parameters:", grid_search.best_params_)
    print("Best estimator:", grid_search.best_estimator_)
    print("Best cross-validation score(accuracy)): {:.5f}".format(grid_search.best_score_))
    print("Test set AUC: {:.5f}".format(roc_auc_score(y_test, grid_search.decision_function(X_test))))
    print("Test set score(accuracy): {:.5f}".format(grid_search.score(X_test, y_test)))
    print("Best estimator's accuracy of test set: {:.5f}".format(grid_search.best_estimator_.score(X_test, y_test)))

    # 使用 AUC 评分
    grid_search = GridSearchCV(SVC(gamma='auto'), param_grid=param_grid, scoring='roc_auc', cv=5)
    grid_search.fit(X_train, y_train)
    show_subtitle(f"网格搜索的评估指标：{grid_search.scoring}")
    print("Best parameters:", grid_search.best_params_)
    print("Best estimator:", grid_search.best_estimator_)
    print("Best cross-validation score(AUC): {:.5f}".format(grid_search.best_score_))
    print("Test set AUC: {:.5f}".format(roc_auc_score(y_test, grid_search.decision_function(X_test))))
    print("Test set score(AUC): {:.5f}".format(grid_search.score(X_test, y_test)))
    print("Best estimator's accuracy of test set: {:.5f}".format(grid_search.best_estimator_.score(X_test, y_test)))

    from sklearn.metrics.scorer import SCORERS
    show_title("系统提供的有效的评估指标")
    print("Available scorers:")
    print(sorted(SCORERS.keys()))


def main():
    # 不平衡的数据对模型的影响
    # evaluate_imbalanced_dataset()
    # 使用混淆矩阵分析不平衡的数据对模型的影响
    # compare_confusion_matrix()
    # 图形显示混淆矩阵的定义
    # show_confusion_matrix()
    # 使用F1分数分析不平衡的数据对模型的影响
    # compare_f1_score()
    # 使用分类报告分析不平衡的数据对模型的影响
    # compare_classification_report()
    # 不平衡的二分类问题
    # imbalanced_two_classes()
    # 5. 准确率——召回率曲线
    # compare_precision_recall_curve()
    # 6. 受试者工作特征（ROC）与 AUC
    # compare_roc_curve()
    # 对比 SVC 的不同 gamma 值条件下的 ROC 曲线，从而选择最优参数配置
    # compare_svc_gamma_roc()
    # 5.3.3 多分类指标
    # multi_classes_score()
    # 5.3.5 在模型选择中使用评估指标
    # model_selection_with_score()
    pass


if __name__ == "__main__":
    main()
    beep_end()
    show_figures()
