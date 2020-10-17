# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C0305_clustering.py
@Version    :   v0.1
@Time       :   2019-10-16 11:37
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Sec02?
@Desc       :   监督学习算法。
"""
from datasets.load_data import load_people, load_train_test_faces
from tools import *


# 3.5. 聚类（clustering）
# 聚类：将数据集划分成组的任务，这些组叫做簇（cluster）。目标是划分数据，使得一个簇内的数据点非常相似且不同簇内的数据点非常不同。
# 3.5.1 K均值聚类
# K均值聚类：最简单、最常用的聚类算法之一。试图找出代表数据特定区域的簇中心。
# 算法步骤：
#   - 随机初始化三个簇中心
#   - 将每个数据点分配给最近的簇中心
#   - 将每个簇中心设置为所分配的所有数据点的平均值
#   - 如果簇的分配不再发生变化，那么算法结束。
def plot_kmeans_algorithm():
    mglearn.plots.plot_kmeans_algorithm()
    plt.suptitle("图3-23：输入数据与K均值算法的三个步骤")


def plot_kmeans_boundaries():
    mglearn.plots.plot_kmeans_boundaries()
    plt.suptitle("图3-24：K均值算法找到的簇中心和簇边界")


def knn_cluster():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs
    X_blobs, y_blobs = make_blobs(random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X_blobs, y_blobs, random_state=seed)

    # 3个簇中心
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)

    # 训练结果
    print('=' * 20)
    print("原始训练数据集的类别标签：\n", y_train)
    print("原始训练数据集的类别标签统计结果：", np.bincount(y_train))
    print('-' * 20)
    print("K均值算法「无监督聚类」训练数据集的输出:\n", kmeans.labels_)
    print("K均值算法「无监督聚类」训练数据集的输出的统计结果", np.bincount(kmeans.labels_))
    print('-' * 20)
    print("原始测试数据集的类别标签：", y_test)
    print("原始测试数据集的类别标签统计结果：", np.bincount(y_test))
    print('-' * 20)
    print("K均值算法「无监督聚类」测试数据集的输出:\n", kmeans.predict(X_test))  # 预测数据
    print("K均值算法「无监督聚类」测试数据集的输出的统计结果", np.bincount(kmeans.predict(X_test)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    # 绘制原始数据点的散点图

    ax = axes[0]
    mglearn.discrete_scatter(x1=X_blobs[:, 0], x2=X_blobs[:, 1], y=y_blobs, markers=['o'], ax=ax)
    ax.set_title("原始数据的散点图")

    # 绘制数据点的散点图
    ax = axes[1]
    mglearn.discrete_scatter(x1=X_train[:, 0], x2=X_train[:, 1], y=kmeans.labels_, markers=['o'], ax=ax)
    # 绘制簇中心的散点图
    mglearn.discrete_scatter(x1=kmeans.cluster_centers_[:, 0],
                             x2=kmeans.cluster_centers_[:, 1],
                             y=[0, 1, 2], markers=['^'], markeredgewidth=2, ax=ax)
    ax.set_title("图3-25：3个簇的K均值算法找到的簇分配和簇中心")

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    plt.suptitle("图3-26：K均值算法找到的簇分配")
    from sklearn.cluster import KMeans
    for cluster in ([2, 3, 4, 5]):
        kmeans = KMeans(n_clusters=cluster)
        kmeans.fit(X_train)
        ax = axes[cluster // 2 - 1][cluster % 2]
        ax.set_title("{}个簇".format(cluster))
        mglearn.discrete_scatter(x1=X_train[:, 0], x2=X_train[:, 1], y=kmeans.labels_, ax=ax)


# 1. K 均值的失败案例
# K均值算法要求每个簇都是凸形（convex）的。
def knn_failed():
    # 生成一些随机分组数据(默认3个中心)
    from sklearn.datasets import make_blobs
    X_varied, y_varied = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5], random_state=170)

    from sklearn.cluster import KMeans
    k_means = KMeans(n_clusters=3, random_state=seed)
    y_predict = k_means.fit_predict(X_varied)

    fig, axes = plt.subplots(2, 1)
    ax = axes[0]
    ax.set_title("原始数据")
    mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_varied, ax=ax)
    ax = axes[1]
    ax.set_title("KNN分类数据")
    mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_predict, ax=ax)
    plt.legend(['cluster 0', 'cluster 1', 'cluster 2'], loc='best')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.suptitle("图3-27：簇的密度不同时，K均值找到的簇分配")

    # 生成一些随机分组数据(默认3个中心)
    from sklearn.datasets import make_blobs
    X_blobs, y_blobs = make_blobs(random_state=170, n_samples=600)

    fig, axes = plt.subplots(2, 2)
    plt.suptitle("图3-28：K均值无法识别非球形簇")

    ax = axes[0, 0]
    ax.set_title("没有变换前的原始数据")
    ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap=mglearn.cm3)

    from sklearn.cluster import KMeans
    k_means = KMeans(n_clusters=3)
    y_predict = k_means.fit_predict(X_blobs)
    ax = axes[1, 0]
    ax.set_title("K均值找到的簇分配")
    ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_predict, cmap=mglearn.cm3)
    ax.scatter(x=k_means.cluster_centers_[:, 0], y=k_means.cluster_centers_[:, 1],
               c=[0, 1, 2], cmap=mglearn.cm3, marker='^', s=100, linewidths=2)

    # 变换数据使其拉长
    rng = np.random.RandomState(74)
    transformation = rng.normal(size=(2, 2))
    X = np.dot(X_blobs, transformation)
    ax = axes[0, 1]
    ax.set_title("变换后的原始数据")
    ax.scatter(X[:, 0], X[:, 1], c=y_blobs, cmap=mglearn.cm3)

    # 将数据聚类成3个簇
    from sklearn.cluster import KMeans
    k_means = KMeans(n_clusters=3)
    y_predict = k_means.fit_predict(X)
    ax = axes[1, 1]
    ax.set_title("K均值找到的簇分配")
    ax.scatter(x=X[:, 0], y=X[:, 1], c=y_predict, cmap=mglearn.cm3)
    ax.scatter(x=k_means.cluster_centers_[:, 0], y=k_means.cluster_centers_[:, 1],
               c=[0, 1, 2], cmap=mglearn.cm3, marker='^', s=100, linewidths=2)

    # 画出簇分配和簇中心
    # fig, axes = plt.subplots(2, 1)
    # plt.suptitle("图3-28：K均值无法识别非球形簇。变换后的原始数据（上）和KNN分类数据（下）")
    # axes[0].scatter(X[:, 0], X[:, 1], c = y_blobs, cmap = mglearn.cm3)
    # axes[1].scatter(x = X[:, 0], y = X[:, 1], c = y_predict, cmap = mglearn.cm3)
    # axes[1].scatter(x = k_means.cluster_centers_[:, 0], y = k_means.cluster_centers_[:, 1],
    #                 c = [1, 2, 0], cmap = mglearn.cm3, marker = '^', s = 100, linewidths = 2)

    # two_moons 数据，K 均值表现很差
    from sklearn.datasets import make_moons
    X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=seed)
    from sklearn.cluster import KMeans
    k_means = KMeans(n_clusters=2)
    k_means.fit(X_moons)
    y_predict = k_means.predict(X_moons)

    fig, axes = plt.subplots(2, 1)
    plt.suptitle("图3-29：K均值无法识别具有复杂形式的簇")
    ax = axes[0]
    ax.set_title("原始数据")
    ax.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap=mglearn.cm2, s=60)
    ax = axes[1]
    ax.set_title("KNN分类数据")
    ax.scatter(X_moons[:, 0], X_moons[:, 1], c=y_predict, cmap=mglearn.cm2, s=60)
    ax.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1],
               marker='^', c=[0, 1], s=100, linewidths=2, cmap=mglearn.cm2)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')


# 2. 失量量化，也是 K 均值分解
# K均值利用簇中心来表示每个数据点，即用一个分量来表示每个数据点，这个分量由簇中心给出，被称为失量量化（Vector Quantization）
def kmeans_vector_quantization():
    people = load_people()
    image_shape = people.images[0].shape
    X_train, X_test, y_train, y_test = load_train_test_faces()

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=100, random_state=seed)
    kmeans.fit(X_train)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, random_state=seed)
    pca.fit(X_train)
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=100, random_state=seed)
    nmf.fit(X_train)

    fig, axes = plt.subplots(3, 15, figsize=(20, 10), subplot_kw={'xticks': (), 'yticks': ()})
    fig.suptitle('图3-30：对比K均值的簇中心与PCA和NMF找到的分量')
    # K均值找到的是图片的共性，PCA 找到的是图片变化最大的特征，NMF 找到的是图片中基础元素
    for ax, comp_kmeans, comp_pca, comp_nmf in zip(
            axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
        ax[0].imshow(comp_kmeans.reshape(image_shape))
        ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
        ax[2].imshow(comp_nmf.reshape(image_shape))

    axes[0, 0].set_ylabel('kmeans')
    axes[1, 0].set_ylabel('pca')
    axes[2, 0].set_ylabel('nmf')

    X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
    X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
    X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)

    fig, axes = plt.subplots(4, 5, figsize=(20, 10), subplot_kw={'xticks': (), 'yticks': ()})
    fig.suptitle('图3-31：利用100个分量（或簇中心）的K均值、PCA和NMF的图像重建的对比——K均值的每张图像仅使用了一个簇中心')
    # K均值重建的效果没有其他两种好，因为K均值取的是平均值，而每个类别有自己的特点，而不是所有特点的平均
    for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(
            axes.T, X_test, X_reconstructed_kmeans, X_reconstructed_pca, X_reconstructed_nmf):
        ax[0].imshow(orig.reshape(image_shape))
        ax[1].imshow(rec_kmeans.reshape(image_shape))
        ax[2].imshow(rec_pca.reshape(image_shape))
        ax[3].imshow(rec_nmf.reshape(image_shape))

    axes[0, 0].set_ylabel('original')
    axes[1, 0].set_ylabel('kmeans')
    axes[2, 0].set_ylabel('pca')
    axes[3, 0].set_ylabel('nmf')


def kmeans_two_moons():
    # 使用K 均值处理two_moons数据
    from sklearn.datasets import make_moons
    X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=seed)
    plt.figure()
    plt.suptitle("原始数据")
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, s=60, cmap='Paired')
    from sklearn.cluster import KMeans
    # for cluster_number in [10]:
    for cluster_number in range(2, 12, 3):
        kmeans = KMeans(n_clusters=cluster_number, random_state=seed)
        kmeans.fit(X_moons)
        y_predict = kmeans.predict(X_moons)
        print('Cluster memberships:\n{}'.format(y_predict))

        # 将2维数据变换到10维数据，每个点到10个簇中心的距离是其特征
        distance_features = kmeans.transform(X_moons)
        print('-' * 20)
        print("Moons shape: {}".format(X_moons.shape))
        print('Distance feature shape: {}'.format(distance_features.shape))
        print('Distance features:\n{}'.format(distance_features))

        plt.figure()
        plt.suptitle("图3-32：利用K均值的{}个簇来表示复杂数据集中的变化".format(cluster_number))
        plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_predict, s=60, cmap='Paired')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c=range(kmeans.n_clusters),
                    s=60, cmap='Paired', marker='^', linewidths=2)
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')

        plt.figure()
        plt.suptitle("图3-32：利用K均值的{}个簇来恢复复杂数据集中的变化".format(cluster_number))
        plt.scatter(distance_features[:, 0], distance_features[:, 1], c=y_predict, s=60, cmap='Paired')
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        pass


# 3.5.2. 凝聚聚类：基于相同原则构建的聚类算法。
# 依据链接准则，将度量“最相似的簇”合并，直到簇的个数满足停止准则。
# scikit-learn使用了以下三种凝聚准则：
#   - ward：默认选项。挑选两个簇合并，使得所有簇的方差增加最小，可以得到大小差不多相等的簇
#   - average：将簇中所有点之间的平均距离最小的两个簇合并
#   - complete：也叫最大链接。将簇中点之间最大距离最小的两个簇合并
def plot_agglomerative_algorithm():
    mglearn.plots.plot_agglomerative_algorithm()
    plt.suptitle("图3-33：凝聚聚类用迭代的方式合并两个最近的簇")


def three_centers_agglomerative():
    from sklearn.datasets import make_blobs
    center = 3
    X_blobs, y_blobs = make_blobs(random_state=1, centers=center)
    plt.figure()
    mglearn.discrete_scatter(X_blobs[:, 0], X_blobs[:, 1], y_blobs)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.suptitle("原始数据")
    from sklearn.cluster import AgglomerativeClustering
    for cluster in [2, 3, 6]:
        agg = AgglomerativeClustering(n_clusters=cluster)
        assignment = agg.fit_predict(X_blobs)

        plt.figure()
        mglearn.discrete_scatter(X_blobs[:, 0], X_blobs[:, 1], assignment)
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.suptitle("图3-34：使用{}个簇对{}个类别的数据进行凝聚聚类的簇分配".format(cluster, center))


def two_moons_agglomerative():
    # 使用凝聚聚类处理two_moons数据
    from sklearn.datasets import make_moons
    X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=seed)
    plt.figure()
    plt.suptitle("原始数据")
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, s=60, cmap='Paired')
    from sklearn.cluster import AgglomerativeClustering
    for cluster in [2, 3, 6]:
        agg = AgglomerativeClustering(n_clusters=cluster)
        assignment = agg.fit_predict(X_moons)

        plt.figure()
        mglearn.discrete_scatter(X_moons[:, 0], X_moons[:, 1], assignment)
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.suptitle("图3-34：使用{}个簇对双月数据集进行凝聚聚类的簇分配".format(cluster))


# 1. 层次聚类与树状图
# 凝聚聚类生成的就是一种层次聚类（Hierarchical Clustering）。
def plot_agglomerative_hierarchical_clustering():
    mglearn.plots.plot_agglomerative()
    plt.suptitle("图3-35：凝聚聚类生成的层次化的簇分配（用线表示）\n以及带有编号的数据点（参见图3-36）")

    # 层次聚类无法展现超过二维的数据，而树状图可以
    # 树状图可以使用SciPy提供的函数来生成，函数接受数据数组，再计算出链接数组
    from sklearn.datasets import make_blobs
    X_blobs, y_blobs = make_blobs(random_state=0, n_samples=12)
    plt.figure()
    # ward()实现聚类
    from scipy.cluster.hierarchy import dendrogram, ward
    linkage_array = ward(X_blobs)
    # dendrogram()绘制树状图
    dendrogram(linkage_array)
    ax = plt.gca()
    bounds = ax.get_xbound()
    ax.plot(bounds, [7.25, 7.25], '--', c='k')
    ax.plot(bounds, [4, 4], '--', c='k')
    # 标注“两个簇”和“三个簇”的位置
    ax.text(bounds[1], 7.25, "两个簇", va='center', fontdict={'size': 10})
    ax.text(bounds[1], 4, "三个簇", va='center', fontdict={'size': 10})
    plt.xlabel('Sample index')
    plt.ylabel('Cluster distance')
    plt.suptitle("图3-36：图3-35中聚类的树状图（用线表示划分成两个簇和三个簇）")


# 3.5.3. DBSCAN 解决 two_moons 数据分类问题，而凝聚聚类不行。
# DBSCAN（Density-Based Spatial Clustering of Applications with Noise），即“具有噪声的基于密度的空间聚类应用”。
# 优点：不需要用户先验地设置簇的个数，可以划分具有复杂开头的簇，可以找出不属于任何簇的点
# 缺点：比凝聚聚类和K均值的速度慢
# 原理：识别特征空间的“拥挤”区域中的点，这些数据点靠在一起的区域称为密集区域。
# 思想：簇由数据的密集区域确定，并由相对较空的区域隔开。
# 在密集区域内的点被称为核心样本（Core Sample），也叫「核心点」。
# 与核心点的距离在eps之内的点称为「边界点」。
# 距离起始点的距离在eps范围内的数据点个数小于min_samples时，这个起始点被称为「噪声点」。
# DBSCAN使用两个参数：min_samples和eps定义。
# 如果在距一个给定数据eps的距离内至少有min_samples个数据点，那么这个数据点就是核心样本，这些核心样本放在一个簇内。
# DBSCAN对访问顺序会有轻度依赖。
def dbscan_blobs():
    from sklearn.datasets import make_blobs
    # 生成三个中心的独立高斯数据集
    # 12个数据点是与后面的图形匹配
    X_blobs, y_blobs = make_blobs(random_state=seed, n_samples=12)
    plt.figure()
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap=mglearn.cm3)
    plt.suptitle("原始数据有12个数据点")

    show_title("默认：eps=0.5, min_samples=5")
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(n_jobs=7)
    clusters = dbscan.fit_predict(X_blobs)
    print('Cluster memberships:\n{}'.format(clusters))
    print("Cluster class number:\n{}".format(np.unique(clusters).size))

    mglearn.plots.plot_dbscan()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("图3-37：DBSCAN找到的簇分配")

    from matplotlib.colors import ListedColormap
    cm_cycle = ListedColormap(['#0000FF', '#FF0000', '#008000', '#000000', '#FFB6C1',
                               '#0000aa', '#ff5050', '#50ff50', '#9040a0', '#fff000'])

    from sklearn.datasets import make_blobs
    # 生成三个中心的独立高斯数据集
    # 120个数据点是与后面的图形匹配
    X_blobs, y_blobs = make_blobs(random_state=seed, n_samples=120)

    # 将数据缩放成均值为0，方差为1，方便设置eps
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_blobs)

    plt.figure()
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_blobs, cmap=mglearn.cm3)
    plt.suptitle("原始数据有120个数据点")

    fig, axes = plt.subplots(3, 5, figsize=(20, 10), subplot_kw={'xticks': (), 'yticks': ()})
    from sklearn.cluster import DBSCAN

    for ax, (min_sample, eps) in zip(axes.ravel(),
                                     [(2, 0.1), (2, 0.3), (2, 0.5), (2, 1.0), (2, 2.0),
                                      (3, 0.1), (3, 0.3), (3, 0.5), (3, 1.0), (3, 2.0),
                                      (5, 0.1), (5, 0.3), (5, 0.5), (5, 1.0), (5, 2.0), ]):
        clusters = DBSCAN(eps=eps, min_samples=min_sample).fit_predict(X_scaled)
        show_title(f"min_sample={min_sample},eps={eps}")
        print("Cluster class number:\n{}".format(np.unique(clusters).size))

        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=cm_cycle, s=60)
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
        ax.set_title("min_samples={}, eps={}".format(min_sample, eps))
    plt.suptitle("图3-38：DBSCAN找到的簇分配")


def dbscan_two_moons():
    from sklearn.datasets import make_moons
    X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=seed)

    # 将数据缩放成均值为0，方差为1，方便设置eps
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_moons)
    X_scaled = scaler.transform(X_moons)

    # red,blue,green,black,pink,
    from matplotlib.colors import ListedColormap
    cm_cycle = ListedColormap(['#0000FF', '#FF0000', '#008000', '#000000', '#FFB6C1',
                               '#0000aa', '#ff5050', '#50ff50', '#9040a0', '#fff000'])

    fig, axes = plt.subplots(3, 4, figsize=(20, 10), subplot_kw={'xticks': (), 'yticks': ()})
    from sklearn.cluster import DBSCAN

    for ax, (min_sample, eps) in zip(axes.ravel(),
                                     [(2, 0.2), (2, 0.3), (2, 0.6), (2, 0.7),
                                      (5, 0.2), (5, 0.3), (5, 0.6), (5, 0.7),
                                      (10, 0.2), (10, 0.3), (10, 0.6), (10, 0.7), ]):
        dbscan = DBSCAN(eps=eps, min_samples=min_sample)
        clusters = dbscan.fit_predict(X_scaled)

        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=cm_cycle, s=60)
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
        ax.set_title("min_samples={}, eps={}".format(min_sample, eps))
    plt.suptitle("图3-38：DBSCAN找到的簇分配")


# 3.5.4. 聚类算法的对比和评估
# 1）用真实值评估聚类：定量的度量，0表示不相关的聚类。
# ARI（Adjusted Rand Index)
# RI = (a + b) / C(n,2) ,
# 分子：属性一致的样本数，即同属于这一类或都不属于这一类。
# a是真实在同一类、预测也在同一类的样本数；
# b是真实在不同类、预测也在不同类的样本数;
# 分母：任意两个样本为一类有多少种组合
# ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
# ARI is a symmetric measure：
#         adjusted_rand_score(a, b) == adjusted_rand_score(b, a)
# https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index
# NMI（Normalized Mutual Information）：对互信息的归一化。
# 聚类算法的评估还可以参考[周志华，2018]，Sec9.3，P199
def evaluate_algorithm_with_ari():
    from sklearn.datasets import make_moons
    X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=seed)

    # 将数据缩放成均值为0，方差为1
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_moons)
    X_scaled = scaler.transform(X_moons)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5), subplot_kw={'xticks': (), 'yticks': ()})

    # 创建一个随机的簇分配，作为参考
    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(X_moons))

    axes[0, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
    from sklearn.metrics.cluster import adjusted_rand_score
    axes[0, 0].set_title("随机分配\n ARI: {:.2f}".format(
        adjusted_rand_score(y_moons, random_clusters)))

    axes[1, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
    from sklearn.metrics.cluster import normalized_mutual_info_score
    axes[1, 0].set_title('随机分配\n NMI: {:.2f}'.format(
        normalized_mutual_info_score(y_moons, random_clusters, average_method='arithmetic')))

    # 默认：eps=0.5, min_samples=5
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import AgglomerativeClustering
    algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2, linkage="ward"), DBSCAN()]
    for ax_index, algorithm in zip([1, 2, 3], algorithms):
        clusters = algorithm.fit_predict(X_scaled)
        axes[0, ax_index].scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
        axes[0, ax_index].set_title('{}\n ARI: {:.2f}'.format(
            algorithm.__class__.__name__,
            adjusted_rand_score(y_moons, clusters)))

        axes[1, ax_index].scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
        axes[1, ax_index].set_title('{}\n NMI: {:.2f}'.format(
            algorithm.__class__.__name__,
            normalized_mutual_info_score(y_moons, clusters, average_method='arithmetic')))
        pass
    plt.suptitle("图3-39：基于AR对two_moons数据集上算法进行评价")
    # plt.suptitle("图3-39：利用监督ARI分数在two_moons数据集上比较随机分配、K均值、凝聚聚类和DBSCAN")


def compare_ari_nmi():
    # accuracy_score() 是用于精确匹配，而聚类关注的是数据是否在相同的簇
    clusters = [
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    from sklearn.metrics import accuracy_score
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics.cluster import normalized_mutual_info_score

    # 类别标签不匹配，但是分类正确时，度量结果也是最佳分类值
    for i in range(0, 4):
        clusters1 = clusters[i]
        for j in range(i + 1, 4):
            clusters2 = clusters[j]
            print('-' * 20)
            print('clusters1=', clusters1)
            print('clusters2=', clusters2)
            print('Accuracy: {:.2f}'.format(accuracy_score(clusters1, clusters2)))
            print('ARI: {:.2f}'.format(adjusted_rand_score(clusters1, clusters2)))
            print('NMI: {:.2f}'.format(normalized_mutual_info_score(clusters1, clusters2, average_method='geometric')))


# 2）在没有真实值的情况下评估聚类
# （轮廓系数：计算簇的紧致度，越大越好，满分为1，要求簇的形状不能过于复杂）
# 实际效果不好，例如：DBSCAN的分类得分却低于KMeans。
# 基于鲁棒性的聚类指标：先向数据中添加一些噪声，或者使用不同的参数设定，然后运行算法，并对结果进行比较。
# 基于鲁棒性的聚类指标的思想：如果许多算法参数和许多数据扰动返回相同的结果，那么这个指标可能是可信的。Scikit-Learn还未实现。
def evaluate_algorithms_with_silhouette_coefficient():
    # 将数据缩放成均值为0，方差为1
    from sklearn.datasets import make_moons
    X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=seed)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_moons)
    X_scaled = scaler.transform(X_moons)

    fig, axes = plt.subplots(1, 4, figsize=(10, 5), subplot_kw={'xticks': (), 'yticks': ()})

    # 创建一个随机的簇分配，作为参考
    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(X_moons))

    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
    from sklearn.metrics.cluster import silhouette_score
    axes[0].set_title('随机分配: {:.2f}'.format(silhouette_score(X_scaled, random_clusters)))

    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import AgglomerativeClustering
    algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]
    for ax, algorithm in zip(axes[1:], algorithms):
        clusters = algorithm.fit_predict(X_scaled)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
        ax.set_title('{} : {:.2f}'.format(algorithm.__class__.__name__, silhouette_score(X_scaled, clusters)))
        pass
    plt.suptitle("图3-40：基于轮廓分数对two_moons数据集上的算法进行评价")
    # plt.suptitle("图3-40：利用无监督的轮廓分数在two_moons数据集上比较随机分配、K均值、凝聚聚类和DBSCAN"
    #              "\n更符合直觉的DBSCAN的轮廓分数低于K均值找到的分配")


# 3）在人脸数据集上比较算法
# 无法保证聚类是按照人所需要的语义要求分类的，因此唯一的办法就是人工分析
# DBSCAN 只是帮助找出异常数据，
# 但是从人脸数据集的选择中可以看出，异常数据不代表错误数据，只是不能在数据集中找到共性，
# 异常数据某些时候可能代表的正是数据信息量最大的数据
def show_dbscan_noise_faces():
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
    image_shape = people.images[0].shape
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    X_people = X_people / 255.

    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, whiten=True, random_state=seed)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)

    # np.unique([0, 1, 2, 3, 0])，只留唯一的数字

    from sklearn.cluster import DBSCAN

    print('=' * 20)
    print("-- 默认参数(eps=0.5, min_samples=5)，返回的标签都是-1，即所有数据都被标为噪声 --")
    dbscan = DBSCAN()
    labels = dbscan.fit_predict(X_pca)
    print('Unique labels: {}'.format(np.unique(labels)))

    print('=' * 20)
    print("-- 设置参数(eps=0.5, min_samples=3)，返回的标签都是-1，即所有数据都被标为噪声 --")
    dbscan = DBSCAN(min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    print('Unique labels: {}'.format(np.unique(labels)))

    print('=' * 20)
    print("-- 设置参数(eps=15, min_samples=3)，返回的标签有-1和0，即部分数据被标为噪声")
    dbscan = DBSCAN(min_samples=3, eps=15)
    labels = dbscan.fit_predict(X_pca)
    labels_list = np.unique(labels)
    for i in range(len(labels_list)):
        print('-' * 20)
        print('Unique labels: {}'.format(labels_list[i]))
        print('Number of points per cluster: {}'.format(np.bincount(labels + 1)[i]))

    noise = X_people[labels == -1]
    cluster = X_people[labels == 0]

    fig, axes = plt.subplots(3, 9, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for image, ax in zip(noise, axes.ravel()):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        pass
    plt.suptitle("图3-41：人脸数据集中26个被DBSCAN标记为噪声的样本")

    fig, axes = plt.subplots(3, 9, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for image, ax in zip(cluster, axes.ravel()):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        pass
    plt.suptitle("显示前27个标记为数据的样本")


def evaluate_dbscan_in_faces():
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    X_people = X_people / 255.

    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, whiten=True, random_state=seed)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)

    # 不同的eps的效果
    from sklearn.cluster import DBSCAN
    for eps in [1, 3, 5, 7, 9, 11, 13]:
        print('-' * 20)
        print('eps={}'.format(eps))
        dbscan = DBSCAN(eps=eps, min_samples=2)
        labels = dbscan.fit_predict(X_pca)

        label_list = np.unique(labels)
        label_count = np.bincount(labels + 1)
        print('Clusters present: {}'.format(label_list))
        print('Clusters size: {}'.format(label_count))
        pass
    pass


def show_eps_7_images():
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
    image_shape = people.images[0].shape
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    y_people = people.target[mask]
    X_people = X_people / 255.

    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, whiten=True, random_state=seed)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)

    # eps=7的时候，看看聚类的效果
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(min_samples=3, eps=7)
    labels = dbscan.fit_predict(X_pca)

    for cluster in range(max(labels) + 1):
        mask = labels == cluster
        n_images = int(np.sum(mask))
        fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4), subplot_kw={'xticks': (), 'yticks': ()})
        for image, label, ax in zip(X_people[mask], y_people[mask], axes):
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1])
            pass
        pass
    plt.suptitle("图3-42：eps=7的DBSCAN找到的簇")


# 用K 均值分析人脸数据集
# K均值可以创建更加均匀大小的簇。
# 降维的PCA可以增加图像的平滑度
def evaluate_kmeans_in_faces():
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
    image_shape = people.images[0].shape
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    y_people = people.target[mask]
    X_people = X_people / 255.

    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, whiten=True, random_state=seed)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=30, random_state=seed)  # 不同的簇中心数目可以得到更加平滑的聚类效果
    labels_km = km.fit_predict(X_pca)
    print('K-Means聚类簇的大小: {}'.format(np.bincount(labels_km)))

    fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for center, ax in zip(km.cluster_centers_, axes.ravel()):
        ax.imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1)
    plt.suptitle("图3-43：将簇的数量设置为10时，K均值找到的簇中心")

    mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people, y_people, people.target_names)
    plt.suptitle("图3-44：K均值为每簇找到的样本图像\n簇中心在左边，5个距中心最近的点，5个距中心最远的点")


# 用凝聚聚类分析人脸数据集
# 凝聚聚类生成的也是大小相近的簇，比DBSCAN更加均匀，没有K均值均匀。
def evaluate_agglomerative_in_faces():
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
    image_shape = people.images[0].shape
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    y_people = people.target[mask]
    X_people = X_people / 255.

    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, whiten=True, random_state=seed)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=10, random_state=seed)  # 不同的簇中心数目可以得到更加平滑的聚类效果
    labels_km = km.fit_predict(X_pca)

    from sklearn.cluster import AgglomerativeClustering
    agglomerative = AgglomerativeClustering(n_clusters=10)
    labels_agg = agglomerative.fit_predict(X_pca)
    print('=' * 20)
    print("-- AgglomerativeClustering(n_clusters = 10) --")
    print('凝聚聚类簇的大小: {}'.format(np.bincount(labels_agg)))
    print('=' * 20)
    print("-- 使用ARI来度量凝聚聚类和K均值给出的两种数据划分的相似度 --")
    from sklearn.metrics.cluster import adjusted_rand_score
    print('ARI: {:.2f}'.format(adjusted_rand_score(labels_agg, labels_km)))
    print("结论：两种聚类的共同点很少。")

    from scipy.cluster.hierarchy import dendrogram, ward
    linkage_array = ward(X_pca)
    plt.figure(figsize=(20, 5))
    # 树状图已经被限制了树的深度，因为显示所有的数据点，图形就无法阅读了。
    dendrogram(linkage_array, p=7, truncate_mode='level', no_labels=True)
    plt.xlabel('样本编号')
    plt.ylabel('簇的距离')
    plt.suptitle("图3-45：凝聚聚类在人脸数据集上的树状图")

    n_clusters = 10
    fig, axes = plt.subplots(n_clusters, 10, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for cluster in range(n_clusters):
        mask = labels_agg == cluster
        axes[cluster, 0].set_ylabel(np.sum(mask))
        for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes[cluster]):
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1], fontdict={'fontsize': 9})
            pass
        plt.suptitle("图3-46：凝聚聚类生成的簇中的随机图像\n每一行对应一个簇，左侧的数字表示每个簇中图像的数量")
        plt.tight_layout()
        pass

    n_clusters = 40
    from sklearn.cluster import AgglomerativeClustering
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agg = agglomerative.fit_predict(X_pca)
    print('=' * 20)
    print("-- AgglomerativeClustering(n_clusters = 40) --")
    print('凝聚聚类簇的大小: {}'.format(np.bincount(labels_agg)))

    fig, axes = plt.subplots(5, 15, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for i, cluster in enumerate([10, 13, 19, 22, 36]):
        mask = labels_agg == cluster
        cluster_size = np.sum(mask)
        axes[i, 0].set_ylabel('# {}: {}'.format(cluster, cluster_size))
        for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes[i]):
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1], fontdict={'fontsize': 9})
            pass
        plt.suptitle("图3-46：凝聚聚类生成的簇中的随机图像\n每一行对应一个簇，左侧的数字表示每个簇中图像的数量")
        pass


def main():
    # 图3-23：输入数据与K均值算法的三个步骤
    # plot_kmeans_algorithm()
    # 图3-24：K均值算法找到的簇中心和簇边界
    # plot_kmeans_boundaries()
    # KNN聚类的效果
    # 图3-25：3个簇的K均值算法找到的簇分配和簇中心
    # 图3-26：K均值算法找到的簇分配
    # knn_cluster()
    # K均值算法要求每个簇都是凸形（convex）的，否则聚类会失败
    # knn_failed()
    # 对比K均值、PCA 和 NMF 处理失量数据（图片）的效果，K均值的效果不好。
    # kmeans_vector_quantization()
    # 双月数据集使用K均值进行聚类，无法正确处理。
    # kmeans_two_moons()
    # 图3-33：凝聚聚类用迭代的方式合并两个最近的簇
    # plot_agglomerative_algorithm()
    # 三个数据中心的数据集的凝聚聚类
    # three_centers_agglomerative()
    # 双月数据集的凝聚聚类
    # two_moons_agglomerative()
    # 图3-36：图3-35中聚类的树状图（用线表示划分成两个簇和三个簇）
    # plot_agglomerative_hierarchical_clustering()
    # 3.5.3. DBSCAN
    # DBSCAN 不能解决高斯抽样的数据集，因为随机性太强，凝聚性不足
    # dbscan_blobs()
    # DBSCAN 解决 two_moons 数据分类问题，而凝聚聚类不行。
    # dbscan_two_moons()
    # 3.5.4. 聚类算法的对比和评估
    # 专门用于无监督学习中比较类别，不要求精确匹配
    # evaluate_algorithm_with_ari()
    # ARI(adjusted rand index)-调整rand指数
    # NMI（normalized mutual information）-归一化互信息
    # 直观地理解ARI与NMI在相同数据集上的评估结果
    # compare_ari_nmi()
    # 2）在没有真实值的情况下评估聚类
    # （轮廓系数：计算簇的紧致度，越大越好，满分为1，要求簇的形状不能过于复杂）
    # evaluate_algorithms_with_silhouette_coefficient()
    # 基于人脸数据集，DBSCAN 设置不同参数，得到不同结果的对比
    # show_dbscan_noise_faces()
    # 使用 DBSCAN 分析人脸数据集，DBSCAN 设置不同的eps的效果对比。效果不好，分析原因可能是没有提供合适的特征，只使用了图片的原始特征。
    # evaluate_dbscan_in_faces()
    # 用K 均值分析人脸数据集：K均值可以创建更加均匀大小的簇。
    # evaluate_kmeans_in_faces()
    # 用凝聚聚类分析人脸数据集：凝聚聚类生成的也是大小相近的簇，比DBSCAN更加均匀，没有K均值均匀。
    # evaluate_agglomerative_in_faces()
    pass


if __name__ == "__main__":
    main()
    beep_end()
    show_figures()
