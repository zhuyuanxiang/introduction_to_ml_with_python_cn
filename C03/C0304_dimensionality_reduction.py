# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C0304_dimensionality_reduction.py
@Version    :   v0.1
@Time       :   2019-10-06 08:34
@License    :   (C)Copyright 2019-2020, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Ch03, P107
@Desc       :   无监督学习算法。降维
"""
import matplotlib.pyplot as plt
import mglearn
import numpy as np

from config import seed
# 3.4. 降维、特征提取与流形学习
# 3.4.1. 主成分分析：旋转数据集的方法，旋转后的特征之间在统计上不相关。
# 按照方差的大小，顺序选择成分。
from tools import beep_end
from tools import show_figures


def plot_pca_illustration():
    mglearn.plots.plot_pca_illustration()
    plt.suptitle("图3-3：用PCA做数据变换")


# 主成分分析应用的例子：
# 1. 对cancer数据集应用PCA
def feature_histogram_cancer():
    # 导入数据
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    malignant = cancer.data[cancer.target == 0]
    benign = cancer.data[cancer.target == 1]

    # 乳腺癌数据的特征直方图，可以发现许多数据没有可分性
    fig, axes = plt.subplots(5, 2, figsize=(10, 10))
    ax = axes.ravel()
    for i in range(10):
        _, bins = np.histogram(cancer.data[:, i], bins=50)
        print('-' * 20)
        print(bins)
        ax[i].hist(malignant[:, i], bins=bins, color='blue', alpha=.5)
        ax[i].hist(benign[:, i], bins=bins, color='red', alpha=.5)
        ax[i].set_title(cancer.feature_names[i])
        # ax[i].set_yticks((0,100))
    ax[0].set_xlabel('特征量纲')
    ax[0].set_ylabel('频率')
    ax[0].legend(['malignant', 'benign'], loc='best')
    fig.tight_layout()
    plt.suptitle("图3-4：Cancer 数据集中每个类别的特征直方图")


def pca_cancer_standard_scaler_2d():
    """使用PCA方法取出两个主要特征，显示出线性可分性"""
    # 第一个主成分中，所有特征的符号相同，说明所有特征之间存在着相关性
    # 第二个主成分中，所有特征的符号有正有负。
    # 导入数据
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()

    # 使用StandardScaler缩放数据后PCA的成分展示
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(cancer.data)
    cancer_scaled = scaler.transform(cancer.data)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(cancer_scaled)
    cancer_pca = pca.transform(cancer_scaled)

    print('=' * 20)
    print('Cancer 数据集中原始数据的形状: {}'.format(str(cancer_scaled.shape)))
    print('Cancer 数据集中降维数据的形状: {}'.format(str(cancer_pca.shape)))

    plt.figure(figsize=(8, 8))
    mglearn.discrete_scatter(cancer_pca[:, 0], cancer_pca[:, 1], cancer.target)
    plt.legend(cancer.target_names, loc='best')
    plt.gca().set_aspect('equal')
    plt.xlabel('第一个主成分')
    plt.ylabel('第二个主成分')
    plt.suptitle("图3-5：利用前两个主成分绘制 Cancer 数据集的二维散点图")

    print('-' * 20)
    print("PCA 成分的形状: {}".format(pca.components_.shape))
    print("PCA 成分的内容:")
    print(pca.components_)

    plt.matshow(pca.components_, cmap='viridis')
    plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
    plt.yticks([0, 1], ['第一个主成分', '第二个主成分'])
    plt.xlabel('特征')
    plt.ylabel('主成分')
    plt.colorbar()
    plt.suptitle("图3-6：Cancer 数据集前两个主成分的热图")


def pca_cancer_standard_scaler_3d():
    """使用PCA方法取出三个主要特征，前两个已经显示出线性可分性，第三个特征对于线性可分性没有贡献"""
    # 导入数据
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()

    # 使用StandardScaler缩放数据后PCA的成分展示
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(cancer.data)
    cancer_scaled = scaler.transform(cancer.data)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(cancer_scaled)
    cancer_pca = pca.transform(cancer_scaled)

    print('=' * 20)
    print('Cancer 数据集中原始数据的形状: {}'.format(str(cancer_scaled.shape)))
    print('Cancer 数据集中降维数据的形状: {}'.format(str(cancer_pca.shape)))

    figure = plt.figure(figsize=(8, 8))
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(figure)
    mask = (cancer.target == 0)
    ax.scatter(cancer_pca[mask, 0], cancer_pca[mask, 1], cancer_pca[mask, 2], c='b', cmap=mglearn.cm2, s=60)
    ax.scatter(cancer_pca[~mask, 0], cancer_pca[~mask, 1], cancer_pca[~mask, 2], c='r', cmap=mglearn.cm2, s=60,
               marker='^')
    ax.set_xlabel('第一个主成分')
    ax.set_ylabel('第二个主成分')
    ax.set_zlabel('第三个主成分')
    plt.suptitle("图3-5：利用前三个主成分绘制 Cancer 数据集的三维散点图")
    plt.legend(cancer.target_names, loc='best')
    # plt.gca().set_aspect('equal')

    print('-' * 20)
    print("PCA 成分的形状: {}".format(pca.components_.shape))
    print("PCA 成分的内容:")
    print(pca.components_)

    plt.matshow(pca.components_, cmap='viridis')
    plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
    plt.yticks([0, 1, 2], ['第一个主成分', '第二个主成分', '第三个主成分'])
    plt.xlabel('特征')
    plt.ylabel('主成分')
    plt.colorbar()
    plt.suptitle("图3-6：Cancer 数据集前三个主成分的热图")


# 2. 特征提取的特征脸
# 特征提取是一种数据表示，比原始表示更适合于分析。
# 例如：提取的特征脸是脸部照片中变化最大的地方，即人脸的细节，那些细节代表着光影变化较大的地方，比如：轮廓、皱纹等等
# 某些类别的数据过多，会导致数据偏斜，可以通过限制每个类别的数据量来解决
def show_original_faces():
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)

    print('=' * 20)
    print('人脸数据集中数据形状: {}'.format(people.images.shape))
    print('人脸数据集中类别（人的数目）: {}'.format(len(people.target_names)))

    counts = np.bincount(people.target)
    print("前二十个人的情况。")
    print('{0:25} {1:5}'.format("姓名", "照片数目"))
    for i, (count, name) in enumerate(zip(counts, people.target_names)):
        if i <= 20:
            print('{0:25} {1:5}'.format(name, count))
        # if (i + 1) % 3 == 0:
        #     print()
        pass

    fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for target, image, ax in zip(people.target, people.images, axes.ravel()):
        ax.imshow(image)
        ax.set_title("原始图片：" + people.target_names[target])
        pass
    plt.suptitle("图3-7：来自 Wild 数据集中已经标注的人脸中的一些图像")


def knn_classify_pca_faces():
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)

    # 生成一个全0的mask矩阵
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        # 将每个人的前50条数据设置为1，方便取出
        mask[np.where(people.target == target)[0][:50]] = 1

    X_people = people.data[mask]
    y_people = people.target[mask]
    # 将灰度值缩放到[0,1]之间，而不是[0,255]之间，可以得到更好的数据稳定性
    X_people = X_people / 255.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=seed)

    # 1) 使用KNN训练和测试数据
    # 5655=87*65
    # X_train.shape: (1341, 5655)
    # Test set score of 1-knn: 0.27
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print('=' * 20)
    print("-- 使用KNN训练和测试原始数据 --")
    print('原始数据的形状: {}'.format(X_train.shape))
    print('原始数据经过KNN训练后测试集的精度: {:.2f}'.format(knn.score(X_test, y_test)))

    # 2) 使用KNN训练和测试被PCA白化的数据
    # X_train_pca.shape: (1341, 100)
    # Test set score of 1-nn: 0.35
    # pca.components_.shape: (100, 5655)

    # 白化（whitening）：将主成分缩放到相同的尺度，变换后的结果与使用StandardScaler相同。
    mglearn.plots.plot_pca_whitening()
    plt.suptitle("图3-8：利用PCA对数据进行白化处理")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, whiten=True, random_state=seed)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_pca, y_train)
    print('=' * 20)
    print("-- 使用KNN训练和测试经过PCA白化的数据 --")
    print('PCA主成分的形状: {}'.format(pca.components_.shape))
    print('经过PCA白化的数据的形状: {}'.format(X_train_pca.shape))
    print('PCA白化的数据经过KNN训练后测试集的精度: {:.2f}'.format(knn.score(X_test_pca, y_test)))

    image_shape = people.images[0].shape
    fig, axes = plt.subplots(3, 5, figsize=(20, 10), subplot_kw={'xticks': (), 'yticks': ()})
    for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap='viridis')
        ax.set_title('{}.component'.format((i + 1)))
        pass
    plt.suptitle("图3-9：人脸数据集的前15个主成分的成分向量")

    # 图3-10：人脸照片=Σ_(i=0)^(n) x_i * compoents_i
    # 每张照片就是主成分的加权求和
    mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)
    plt.suptitle("图3-11 利用越来越多的主成分对三张人脸图像进行重建")

    plt.figure()
    # mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
    plt.xlabel('第一个主成分')
    plt.ylabel('第二个主成分')
    plt.suptitle("两个主成分的散点图")


# 3.4.2. NMF.非负矩阵分解
def plot_nmf_illustration():
    mglearn.plots.plot_nmf_illustration()
    plt.suptitle("图3-13 两个分量的非负矩阵分解找到的分量（左）和一个分量的非负矩阵分解找到的分量（右）")


def knn_classify_nmf_faces():
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
    image_shape = people.images[0].shape

    # 生成一个全0的mask矩阵
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        # 将每个人的前50条数据设置为1，方便取出
        mask[np.where(people.target == target)[0][:50]] = 1

    X_people = people.data[mask]
    y_people = people.target[mask]
    # 将灰度值缩放到[0,1]之间，而不是[0,255]之间，可以得到更好的数据稳定性
    X_people = X_people / 255.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=seed)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print('=' * 20)
    print("-- 使用KNN训练和测试原始数据 --")
    print('原始数据的形状: {}'.format(X_train.shape))
    print('原始数据经过KNN训练后测试集的精度: {:.2f}'.format(knn.score(X_test, y_test)))

    # 分量太少，学习的精确度较差
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=100, random_state=seed)
    nmf.fit(X_train)
    X_train_nmf = nmf.transform(X_train)
    X_test_nmf = nmf.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_nmf, y_train)

    print('=' * 20)
    print("-- 使用KNN训练和测试经过NMF处理的数据 --")
    print('NMF成分的形状: {}'.format(nmf.components_.shape))
    print('经过NMF的数据的形状: {}'.format(X_train_nmf.shape))
    print('NMF的数据经过KNN训练后测试集的精度: {:.2f}'.format(knn.score(X_test_nmf, y_test)))
    # ToDo: 经过NMF处理的数据精度没有提高？

    fig, axes = plt.subplots(3, 5, figsize=(20, 10), subplot_kw={'xticks': (), 'yticks': ()})
    plt.suptitle("图3-15 使用15个分量的NMF在人脸数据集上找到的15个分量")
    for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap='viridis')
        ax.set_title('{}.component'.format((i + 1)))
        pass

    components = 3  # 不同分量的图片有一定的共性
    indexes = np.argsort(X_train_nmf[:, components])[::-1]
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), subplot_kw={'xticks': (), 'yticks': ()})
    plt.suptitle("图3-16 第3个分量的系数较大的人脸")
    for i, (index, ax) in enumerate(zip(indexes, axes.ravel())):
        ax.imshow(X_train[index].reshape(image_shape))
        pass

    components = 7  # 不同分量的图片有一定的共性
    indexes = np.argsort(X_train_nmf[:, components])[::-1]
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), subplot_kw={'xticks': (), 'yticks': ()})
    plt.suptitle("图3-16 第7个分量的系数较大的人脸")
    for i, (index, ax) in enumerate(zip(indexes, axes.ravel())):
        ax.imshow(X_train[index].reshape(image_shape))
        pass


def plot_nmf_faces():
    """利用越来越多的非负分量对三张人脸图像进行重建"""
    # 下面这张图计算时间比较长，需要耐心等等
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
    image_shape = people.images[0].shape

    # 生成一个全0的mask矩阵
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        # 将每个人的前50条数据设置为1，方便取出
        mask[np.where(people.target == target)[0][:50]] = 1

    X_people = people.data[mask]
    y_people = people.target[mask]
    # 将灰度值缩放到[0,1]之间，而不是[0,255]之间，可以得到更好的数据稳定性
    X_people = X_people / 255.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=seed)

    mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape=image_shape)
    plt.suptitle("图3-14 利用越来越多的非负分量对三张人脸图像进行重建")


# 对比 NMF 和 PCA 方法在超定盲信号分离上的效果
def blind_source_separation():
    # 原始信号（正弦信号、方波信号、锯齿信号）
    # NMF可以比较好的还原原始信号
    # PCA无法有效还原原始信号，因为PCA把最大的方差放在一个还原信号里面
    # 关于分解方法的介绍可以参考以下网址：
    # https://scikit-learn.org/stable/modules/decomposition.html
    from mglearn.datasets import make_signals
    source = mglearn.datasets.make_signals()
    print('原始信号的形状: {}'.format(source.shape))
    plt.figure(figsize=(12, 4))
    plt.plot(source, '-')
    plt.xlabel('时间')
    plt.ylabel('信号')
    plt.suptitle("图3-18 原始信号源")

    # 混合数据（混合成100维的数据）
    A = np.random.RandomState(0).uniform(size=(100, 3))
    X = np.dot(source, A.T)
    print('混合信号的形状: {}'.format(X.shape))

    # 解混数据
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=3, random_state=seed)
    nmf_signal = nmf.fit_transform(X)
    print('NMF 恢复信号的形状: {}'.format(nmf_signal.shape))
    plt.figure(figsize=(12, 4))
    plt.plot(nmf_signal, '-')
    plt.xlabel('时间')
    plt.ylabel('信号')
    plt.suptitle("NMF 变换后的信号源")

    print("NMF 成分的形状: {}".format(nmf.components_.shape))
    plt.figure(figsize=(12, 4))
    x_scale = [n for n in range(0, 100)]
    plt.plot(x_scale, nmf.components_[0], '-')
    plt.plot(x_scale, nmf.components_[1], '-')
    plt.plot(x_scale, nmf.components_[2], '-')
    plt.xlabel('时间')
    plt.ylabel('信号')
    plt.suptitle("NMF 变换后的成分")

    # 使用PCA进行对比
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca_signal = pca.fit_transform(X)
    print('PCA 恢复信号的形状: {}'.format(pca_signal.shape))
    plt.figure(figsize=(12, 4))
    plt.plot(pca_signal, '-')
    plt.xlabel('时间')
    plt.ylabel('信号')
    plt.suptitle("NMF 变换后的信号源")

    print("PCA 成分的形状: {}".format(pca.components_.shape))
    plt.figure(figsize=(12, 4))
    x_scale = [n for n in range(0, 100)]
    plt.plot(x_scale, pca.components_[0], '-')
    plt.plot(x_scale, pca.components_[1], '-')
    plt.plot(x_scale, pca.components_[2], '-')
    plt.xlabel('时间')
    plt.ylabel('信号')
    plt.suptitle("PCA 变换后的成分")

    # 显示四种信号：混合信号、原始信号、NMF分解的信号、PCA分解的信号
    models = [X, source, nmf_signal, pca_signal]
    names = ['观测的信号 (100维数据中的前三个)',
             '真实的信号（正弦信号、方波信号、锯齿信号）',
             'NMF 恢复的信号',
             'PCA 恢复的信号']
    fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5},
                             subplot_kw={'xticks': (), 'yticks': ()})
    plt.suptitle("图3-19 利用NMF和PCA还原混合的信号源")
    for model, name, ax in zip(models, names, axes):
        ax.set_title(name)
        ax.plot(model[:, :3], '-')
        pass


# 3.4.3. 用t-SNE进行流形学习
# 流形学习算法（Manifold Learning）：允许复杂的映射，给出更好的可视化。
# 流形学习算法主要用于数据可视化，很少用来生成多于两个的新特征。主要是计算训练数据的新的表示，而不允许变换出新的数据。
# 意味着这些算法不可以用于测试集，只能用于变换训练集的数据。
# 流形学习对于数据分析的探索很有用。找出数据的二维表示，并且尽可能地保持数据点之间的距离。
# t-SNE重点保持距离较近的点的距离不变，而不是保持距离较远的点的距离不变。试图保存距离较近的点的相关信息。
def t_SNE():
    # 载入手写数字的数据
    from sklearn.datasets import load_digits
    digits = load_digits()
    print("手写数字数据集的形状= {}".format(digits.data.shape))  # (1797, 64)
    print("手写数字数据集中图片的形状= {}".format(digits.images.shape))  # (1797, 8, 8)

    fig, axes = plt.subplots(4, 5, figsize=(10, 5), subplot_kw={'xticks': (), 'yticks': ()})
    plt.suptitle("图3-20：digits 数据集的示例图像")
    for ax, img, target in zip(axes.ravel(), digits.images, digits.target):
        ax.imshow(img)
        ax.set_title(target)
        pass

    # 构建一个PCA模型
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(digits.data)  # digits.data提供的是buffer object，估计是用于其他函数访问数据
    digits_pca = pca.transform(digits.data)
    colors = ['red', 'green', 'blue', 'purple', 'pink',
              'black', 'orange', 'cyan', 'yellow', 'lightgrey']

    plt.figure(figsize=(10, 10))
    plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
    plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
    for i in range(len(digits.data)):
        # 绘制散点图，但是不使用散点表示数据，而用具体的数字表示数据
        plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
                 color=colors[digits.target[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel('第一个主成分')
    plt.ylabel('第二个主成分')
    plt.suptitle("图3-21：利用两个主成分绘制digits数据集的散点图")

    # 构建t-SNE模型（基于流形，使距离近的点更近，距离远的点更远，从而增加可分性）
    # t-SNE并不知道类别标签，只是利用原始空间中数据点之间的靠近程度就可以将类别通过无监督算法进行分隔
    from sklearn.manifold import TSNE
    tsne = TSNE(random_state=seed)
    # 使用fit_transform()函数，是因为t-SNE模型没有transform()函数，因为模型只对训练数据集进行计算，不能再对测试数据集进行计算
    digits_tsne = tsne.fit_transform(digits.data)
    print("经过 t-SNE 变换后的数据的状态={}".format(digits_tsne.shape))

    plt.figure(figsize=(10, 10))
    plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())
    plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max())
    for i in range(len(digits.data)):
        # 绘制散点图，但是不使用散点表示数据，而用具体的数字表示数据
        plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
                 color=colors[digits.target[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel('t-SNE feature 0')
    plt.ylabel('t-SNE feature 1')
    plt.suptitle("图3-22：利用t-SNE找到的两个分量绘制digits数据集的散点图")


if __name__ == "__main__":
    # 图3-3：用PCA做数据变换
    # plot_pca_illustration()

    # 图3-4：Cancer 数据集中每个类别的特征直方图
    # feature_histogram_cancer()
    # 使用PCA方法取出两个主要特征，显示出线性可分性
    # pca_cancer_standard_scaler_2d()

    # 使用PCA方法取出三个主要特征，显示出线性可分性，也显示出第一个主成分作用最大
    pca_cancer_standard_scaler_3d()

    # 图3-7：来自 Wild 数据集中已经标注的人脸中的一些图像
    # show_original_faces()

    # 使用PCA对人脸数据白化处理，提取成分，再利用KNN进行学习发现精度有所提升。
    # knn_classify_pca_faces()

    # 3.4.2. NMF.非负矩阵分解
    # plot_nmf_illustration()

    # 使用NMF对人脸数据白化处理，提取成分，再利用KNN进行学习发现精度有所提升。
    # 相比PCA提取100个成分，NMF只提取15个成分，但是 NMF 提取100个成分，精度也没有提高
    # PCA的计算速度也比NMF快
    # knn_classify_nmf_faces()

    # 利用越来越多的非负分量对三张人脸图像进行重建
    # 下面这张图计算时间比较长，需要耐心等等
    # plot_nmf_faces()

    # 对比 NMF 和 PCA 方法在超定盲信号分离上的效果
    # blind_source_separation()

    t_SNE()

    beep_end()
    show_figures()
