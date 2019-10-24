# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   C07_text_data.py
@Version    :   v0.1
@Time       :   2019-10-13 10:46
@License    :   (C)Copyright 2019-2020, zYx.Tom
@Reference  :   《Python机器学习基础教程》, Ch07，P250
@Desc       :   处理文本数据
"""
import matplotlib.pyplot as plt
import mglearn
import numpy as np

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 3, suppress = True, threshold = np.inf, linewidth = 200)


# 7.2 示例应用：电影评论的情感分析
# 电影评论数据集
def load_databases():
    number_title = "电影评论数据集"
    print('\n', '-' * 5, number_title, '-' * 5)

    # 解压缩数据后，先删除data/aclImdb/train/下的unsup目录及其文件，这个是用于无监督学习的无标签文档。
    from sklearn.datasets import load_files
    reviews_train = load_files("../data/aclImdb/train/")
    text_train, y_train = reviews_train.data, reviews_train.target
    print("Type of text_train: {}".format(type(text_train)))
    print("Length of text_train: {}".format(len(text_train)))
    print("原始的 text_train[1]:")
    print(text_train[1])

    # 删除数据中与内容无关的部分。例如：HTML换行符（<br />）。
    text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
    print('-' * 50)
    print("处理过的 text_train[1]:")
    print(text_train[1])

    print('-' * 50)
    print("训练数据集 中样本的类别: {}".format(np.unique(y_train)))
    print("训练数据集 中每个类别的数目： {}".format(np.bincount(y_train)))

    reviews_test = load_files("../data/aclImdb/test/")
    text_test, y_test = reviews_test.data, reviews_test.target
    print('=' * 50)
    print("Type of text_test: {}".format(type(text_test)))
    print("Length of text_test: {}".format(len(text_test)))
    print("原始的 text_test[1]:")
    print(text_test[1])

    # 测试数据不需要处理，结果也是一样的，估计词袋中没有的词就直接放弃了
    text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
    print('-' * 50)
    print("处理过的 text_test[1]:")
    print(text_test[1])

    print('-' * 50)
    print("测试数据集 中样本的类别： {}".format(np.unique(y_test)))
    print("测试数据集 中每个类别的数目： {}".format(np.bincount(y_test)))


def load_data():
    from sklearn.datasets import load_files
    reviews_train = load_files("../data/aclImdb/train/")
    text_train, y_train = reviews_train.data, reviews_train.target
    reviews_test = load_files("../data/aclImdb/test/")
    text_test, y_test = reviews_test.data, reviews_test.target
    # 删除数据中与内容无关的部分。例如：HTML换行符（<br />）。
    text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
    return text_train, y_train, text_test, y_test


def show_vectorizer_feature(X_train, feature_names):
    print('=' * 50)
    print("训练数据集中的特征个数（词袋数目）: {}".format(len(feature_names)))
    print("First 20 features:")
    print(feature_names[:20])
    print("Features 20010 to 20030:")
    print(feature_names[20010:20030])
    print("Every 2000th feature:")
    print(feature_names[::2000])


def grid_search_logistic_regression(X_train, y_train, X_test, y_test, classifier = None, param_grid = None, cv = 5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    if classifier is None:
        classifier = LogisticRegression(solver = 'lbfgs', max_iter = 10000)
    if param_grid is None:
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

    grid_search = GridSearchCV(classifier, param_grid, cv = cv, n_jobs = 3, iid = True)
    grid_search.fit(X_train, y_train)
    print('=' * 50)
    print("基于网格搜索的 LogisticRegression 模型学习的最佳交叉验证的得分: {:.2f}".format(grid_search.best_score_))
    print("基于网格搜索的 LogisticRegression 模型学习的最佳交叉验证的参数: ", grid_search.best_params_)
    print('-' * 50)
    print("基于网格搜索的 LogisticRegression 模型学习的测试集上的得分：{:.2f}".format(grid_search.score(X_test, y_test)))
    return grid_search


def cross_validation_logistic_regression(X_train, y_train):
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    scores = cross_val_score(LogisticRegression(solver = 'lbfgs', max_iter = 10000),
                             X_train, y_train, cv = 5, n_jobs = 3)
    print('=' * 50)
    print("基于交叉验证的 LogisticRegression 模型学习的平均精确度: {:.2f}".format(np.mean(scores)))
    return LogisticRegression


# 7.3 将文本数据转换为数值表示（词袋）
def transform_to_bag_of_words():
    number_title = "将文本数据转换为数值表示（词袋）"
    print('\n', '-' * 5, number_title, '-' * 5)

    bards_words = ["The fool doth think he is wise", "but the wise man knows himself to be a fool"]

    # 构建单词表
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(bards_words)
    print("原始数据：", bards_words)
    print("单词表的大小: {}".format(len(vectorizer.vocabulary_)))
    print("单词表的内容: {}".format(vectorizer.vocabulary_))
    print("单词表的排序后的内容: {}".format(sorted(vectorizer.vocabulary_)))

    # bag-of-words 表示保存在一个SciPy的稀疏矩阵中。
    print('=' * 50)
    bag_of_words = vectorizer.transform(bards_words)
    print("bag_of_words: {}")
    print(repr(bag_of_words))
    print("Dense representation of bag_of_words: {}")
    print(bag_of_words.toarray())


# 7.3.2 将词袋应用于电影评论
def bag_of_words_for_movie_reviews():
    number_title = "将词袋应用于电影评论"
    print('\n', '-' * 5, number_title, '-' * 5)

    # 运行时间较长
    text_train, y_train, text_test, y_test = load_data()

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(text_train)
    feature_names = vectorizer.get_feature_names()
    X_train = vectorizer.transform(text_train)
    X_test = vectorizer.transform(text_test)
    print('=' * 50)
    print("训练数据集：", repr(X_train))
    # <15000x59994 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 2055222 stored elements in Compressed Sparse Row format>

    show_vectorizer_feature(X_train, feature_names)
    cross_validation_logistic_regression(X_train, y_train)
    grid_search_logistic_regression(X_train, y_train, X_test, y_test)
    # 0.88， 0.88， 0.87


# 删除那些不具有信息量的单词，即某些单词只出现在少量文档中，例如：数字、错误拼写的单词、生僻词等等
def bag_of_words_for_movie_reviews_min_df():
    number_title = "删除 不具有信息量的单词"
    print('\n', '-' * 5, number_title, '-' * 5)

    # 至少出现在(min_df)5个文档的单词才被选择为特征
    # 数字的个数明显减少，某些生僻词或者拼写错误的词也都消失了。
    text_train, y_train, text_test, y_test = load_data()

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(min_df = 5)  # 至少出现在(min_df)5个文档的单词才被选择为特征
    vectorizer.fit(text_train)
    feature_names = vectorizer.get_feature_names()
    X_train = vectorizer.transform(text_train)
    X_test = vectorizer.transform(text_test)
    print('=' * 50)
    print("删除 不具有信息量的单词(min_df = 5)的训练数据集：", repr(X_train))
    # <15000x21196 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 1991496 stored elements in Compressed Sparse Row format>

    show_vectorizer_feature(X_train, feature_names)
    cross_validation_logistic_regression(X_train, y_train)
    grid_search_logistic_regression(X_train, y_train, X_test, y_test)
    # 0.88，0.88，0.87


def bag_of_words_for_movie_reviews_stop_words():
    number_title = "删除 不具有信息量的单词+停用词"
    print('\n', '-' * 5, number_title, '-' * 5)

    # 至少出现在(min_df)5个文档的单词才被选择为特征
    # 数字的个数明显减少，某些生僻词或者拼写错误的词也都消失了。
    text_train, y_train, text_test, y_test = load_data()

    # 停用词的数目并不多（318个），对于特征维数的降低帮助不大，并且删除后会降低模型精度，
    # 但是可以显著提高计算速度。
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    print('=' * 50)
    print("停用词的个数： {}".format(len(ENGLISH_STOP_WORDS)))
    print("Every 10th stop-word:", list(ENGLISH_STOP_WORDS)[::10])

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(min_df = 5, stop_words = 'english')  # 至少出现在(min_df)5个文档的单词才被选择为特征
    vectorizer.fit(text_train)
    feature_names = vectorizer.get_feature_names()
    X_train = vectorizer.transform(text_train)
    X_test = vectorizer.transform(text_test)
    print('=' * 50)
    print("删除 不具有信息量的单词(min_df = 5)+停用词的训练数据集：", repr(X_train))
    # <15000x20895 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 1270076 stored elements in Compressed Sparse Row format>

    show_vectorizer_feature(X_train, feature_names)
    cross_validation_logistic_regression(X_train, y_train)
    grid_search_logistic_regression(X_train, y_train, X_test, y_test)
    # 0.87，0.88，0.86


def bag_of_words_for_movie_reviews_max_df():
    # 删除那些出现在大量文档中的单词，说明其过于普遍，而不具有足够的区分性，会明显降低模型的精度。
    # 停用词的数目并不多（318个），但是可以显著提高计算速度。
    # 至少出现在(min_df)5个文档的单词才被选择为特征
    # 数字的个数明显减少，某些生僻词或者拼写错误的词也都消失了。
    number_title = "删除 不具有信息量的单词+停用词+出现在大量文档中的单词"
    print('\n', '-' * 5, number_title, '-' * 5)

    # 至少出现在(min_df)5个文档的单词才被选择为特征
    # 数字的个数明显减少，某些生僻词或者拼写错误的词也都消失了。
    text_train, y_train, text_test, y_test = load_data()

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(min_df = 5, stop_words = 'english', max_df = 100)  # 至少出现在(min_df)5个文档的单词才被选择为特征
    vectorizer.fit(text_train)
    feature_names = vectorizer.get_feature_names()
    X_train = vectorizer.transform(text_train)
    X_test = vectorizer.transform(text_test)
    print('=' * 50)
    print("删除 不具有信息量的单词(min_df = 5)+停用词的训练数据集：", repr(X_train))
    # <15000x18613 sparse matrix of type '<class 'numpy.int64'>'
    # 	with 381706 stored elements in Compressed Sparse Row format>
    show_vectorizer_feature(X_train, feature_names)
    cross_validation_logistic_regression(X_train, y_train)

    grid_search_logistic_regression(X_train, y_train, X_test, y_test)
    # 0.79，0.81，0.71
    pass


# 7.5 用tf-idf缩放数据（提高计算的速度，不改变模型的精度）
# Scikit-Learn实现了tf-idf的类：
#   - TfidfTransformer: 接受CountVectorizer生成的稀疏矩阵并将其变换
#   - TfidfVectorizer: 接受文本数据并完成词袋特征提取与tf-idf变换。
def bag_of_words_for_movie_reviews_tf_idf():
    number_title = "删除停用词+使用TF-IDF缩放数据"
    print('\n', '-' * 5, number_title, '-' * 5)

    # 至少出现在(min_df)5个文档的单词才被选择为特征
    # 数字的个数明显减少，某些生僻词或者拼写错误的词也都消失了。
    text_train, y_train, text_test, y_test = load_data()

    from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    pipeline = make_pipeline(
            TfidfVectorizer(min_df = 5, stop_words = 'english'),
            LogisticRegression(solver = 'lbfgs', max_iter = 10000))
    param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}

    grid_search = grid_search_logistic_regression(text_train, y_train, text_test, y_test, pipeline, param_grid)
    # 0.89， 0.86

    vectorizer = grid_search.best_estimator_.named_steps['tfidfvectorizer']
    X_train = vectorizer.transform(text_train)
    print('=' * 50)
    print("TFIDF的训练数据集：", repr(X_train))
    # <15000x20895 sparse matrix of type '<class 'numpy.float64'>'
    # 	with 1270076 stored elements in Compressed Sparse Row format>

    max_value = X_train.max(axis = 0).toarray().ravel()
    feature_names = np.array(vectorizer.get_feature_names())

    sorted_by_tfidf = max_value.argsort()
    print('=' * 50)
    print("Number of features: {}".format(len(feature_names)))  # 20895
    print("Features with lowest tfidf:")
    print(feature_names[sorted_by_tfidf[:20]])
    print("Features with highest tfidf:")
    print(feature_names[sorted_by_tfidf[-20:]])

    sorted_by_idf = np.argsort(vectorizer.idf_)
    print('-' * 50)
    print("Features with lowest idf:")
    print(feature_names[sorted_by_idf[:100]])

    # 7.6 研究模型的系数
    coefficient = grid_search.best_estimator_.named_steps['logisticregression'].coef_
    mglearn.tools.visualize_coefficients(coefficient, feature_names, n_top_features = 40)
    plt.suptitle("图7-2：在TF-IDF特征上训练的Logistic回归的最大系数和最小系数\n"
                 "左侧的负系数属于模型找到的表示负面评论的单词" +
                 '-' * 10 +
                 "右侧的正系数属于模型找到的表示正面评论的单词")


# 7.7 多个单词的词袋（N元分词）
# 在Vectorizer类中设置ngram_range参数即可，参数输入一个元组，表示词例序列的（最小长度，最大长度）。
def transform_to_n_gram():
    number_title = "多个单词的词袋（N元分词）"
    print('\n', '-' * 5, number_title, '-' * 5)

    bards_words = ["The fool doth think he is wise", " but the wise man knows himself to be a fool"]

    # 构建单词表
    from sklearn.feature_extraction.text import CountVectorizer
    for ngram_range in [(1, 1), (2, 2), (1, 3)]:
        print('=' * 50)
        print("ngram_range= {}".format(ngram_range))
        vectorizer = CountVectorizer(ngram_range = ngram_range)
        vectorizer.fit(bards_words)
        print('-' * 50)
        print("单词表的大小: {}".format(len(vectorizer.vocabulary_)))
        print("单词表的内容: {}".format(vectorizer.vocabulary_))
        print("单词表的排序后的内容: {}".format(sorted(vectorizer.vocabulary_)))

        # bag-of-words 表示保存在一个SciPy的稀疏矩阵中。
        print('-' * 50)
        bag_of_words = vectorizer.transform(bards_words)
        print("bag_of_words: {}")
        print(repr(bag_of_words))
        print("Dense representation of bag_of_words: {}")
        print(bag_of_words.toarray())
        pass


# 基于N元语法模型建立电影评论数据集的词袋
def bag_of_words_for_movie_reviews_n_gram():
    # 计算时间非常长。。。
    # 可以将data下的数据减少到每个目录7500个文件，就可以快速看到结果了。
    number_title = "基于N元语法模型建立电影评论数据集的词袋"
    print('\n', '-' * 5, number_title, '-' * 5)

    # 至少出现在(min_df)5个文档的单词才被选择为特征
    # 数字的个数明显减少，某些生僻词或者拼写错误的词也都消失了。
    text_train, y_train, text_test, y_test = load_data()

    from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    pipeline = make_pipeline(
            TfidfVectorizer(min_df = 5, stop_words = 'english'),
            LogisticRegression(solver = 'lbfgs', max_iter = 10000))
    param_grid = {
            'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
            # 'logisticregression__C': [100],
            # 'tfidfvectorizer__ngram_range': [(1, 3)]
            'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)]
    }
    grid_search = grid_search_logistic_regression(text_train, y_train, text_test, y_test, pipeline, param_grid)
    # 0.89, 0.87

    scores = grid_search.cv_results_['mean_test_score'].reshape(-1, 3).T
    heatmap = mglearn.tools.heatmap(
            scores, xlabel = 'C', ylabel = 'ngram_range', cmap = 'viridis', fmt = '%.3f',
            xticklabels = param_grid['logisticregression__C'],
            yticklabels = param_grid['tfidfvectorizer__ngram_range']
    )
    plt.colorbar(heatmap)
    plt.suptitle("图7-3：交叉验证平均精度作为参数ngram_range和C的函数的热图可视化\n"
                 "使用二元分词对精度的提高最明显"
                 "使用三元分词可以得到最佳的精度")

    # # 提取特征的名称和系数
    vectorizer = grid_search.best_estimator_.named_steps['tfidfvectorizer']
    feature_names = np.array(vectorizer.get_feature_names())
    coefficient = grid_search.best_estimator_.named_steps['logisticregression'].coef_
    mglearn.tools.visualize_coefficients(coefficient, feature_names, n_top_features = 40)
    plt.suptitle("图7-4：同时使用TF-IDF缩放与一元分词、二元分词和三元分词时的最重要的特征\n"
                 "左侧的负系数属于模型找到的表示负面评论的单词，"
                 "右侧的正系数属于模型找到的表示正面评论的单词")

    # 三元分词特征可视化
    # ！ 注意：如果系统最佳的是二元分词，那么就得不到结果
    mask = np.array([len(feature.split(" ")) for feature in feature_names]) == 3
    mask_coefficient = coefficient.ravel()[mask]
    mask_feature_names = feature_names[mask]
    if len(mask_feature_names) != 0:
        mglearn.tools.visualize_coefficients(mask_coefficient, mask_feature_names, n_top_features = 40)
        plt.suptitle("图7-5：模型中三元分词的最重要的特征\n"
                     "左侧的负系数属于模型找到的表示负面评论的单词，"
                     "右侧的正系数属于模型找到的表示正面评论的单词")


# 7.8 词干提取和词形还原
def word_normalization():
    # 通过对比会发现词干提取过于粗糙，
    # 会将所有单词不考虑上下文地简化为最简单的形式，例如：所有的meeting都转化为meet，
    # 还会将某些正确单词简化为错误形式，例如：was转化为wa，去年单词最后的s
    # 而词形还原就会更加合理，
    # 例如：两个meeting，前面一个是名词保留，后面一个是动词转化为meet
    # 例如：was和am这类be动词会转换为be
    import spacy, nltk

    number_title = "词干提取(NLTK)和词形还原(SPACY)"
    print('\n', '-' * 5, number_title, '-' * 5)

    # 加载spacy的英语模型
    # python -m spacy download en
    # 我下载时没有使用管理员权限，所以没有创建en的快捷方式，只好使用语言包的命名“en_core_web_sm”
    en_nlp = spacy.load('en_core_web_sm')
    # nltk的Porter词干提取器
    stemmer = nltk.stem.PorterStemmer()

    # 定义函数用于对比spacy中的词形还原与nltk中的词干提取
    def compare_normalization(doc):
        # 在spacy中对文档进行分词
        doc_spacy = en_nlp(doc)
        # 输出spacy还原的词形
        # 1.7.5以上的版本的spacy会把'our'还原成'-PRON-'
        # 注："I'm"在逗号后面必须插入空格，否则无法分析。
        print('=' * 50)
        print(doc)
        print('-' * 50)
        print("Lemmatization:")
        print([token.lemma_ for token in doc_spacy])
        # 输出Porter基于spacy分解的单词提取的词干
        print('-' * 20)
        print("Stemming:")
        print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])
        pass

    # 注意下面在分析“I'm”时的区别。
    print("--> “,”与“I'm”之间必须加空格才能正确分割")

    original_text = u"Our meeting today was worse than yesterday,I'm scared of meeting the clients tomorrow."
    compare_normalization(original_text)

    original_text = u"Our meeting today was worse than yesterday, I'm scared of meeting the clients tomorrow."
    compare_normalization(original_text)
    pass


# 在sklearn中没有实现这两种形式的标准化，但是可以利用一些已有的函数来实现比nltk更好的效果
# 通过使用CountVectorizer所使用的基于正则表达式的分词器来替换spacy的分词器

# 了解正则表达式
def test_regularization():
    number_title = "了解正则表达式"
    print('\n', '-' * 5, number_title, '-' * 5)

    import re
    regexp = re.compile('(?u)\\b\\w\\w+\\b')

    def re_compile_regular(tmp_string):
        print('=' * 50)
        print(tmp_string)
        print('-' * 50)
        print(regexp.findall(tmp_string))
        print()
        pass

    # 过滤了纯数字、单个字母的单词和标点符号。
    tmp_string = u"Our meeting today was worse than yesterday,I'm scared of meeting the clients tomorrow."
    re_compile_regular(tmp_string)
    tmp_string = u"Our meeting today was worse than yesterday, I'm scared of meeting the clients tomorrow."
    re_compile_regular(tmp_string)
    tmp_string = \
        u"Our the 3rd meeting today was worse than yesterday, I am scared of our 3 meetings the clients tomorrow."
    re_compile_regular(tmp_string)
    tmp_string = u"I like a book, a pencil and a rubber."
    re_compile_regular(tmp_string)
    tmp_string = u"We will take a book, open the pencil-box and read the paper."
    re_compile_regular(tmp_string)
    pass


# 在sklearn中没有实现这两种形式的标准化，但是可以利用一些已有的函数来实现比nltk更好的效果
# 通过使用CountVectorizer所使用的基于正则表达式的分词器来替换spacy的分词器
def sklearn_lemmatization():
    # 计算时间非常长，运行前注意！
    # 主要计算时间用在大量单词的词形还原
    number_title = "基于Scikit-Learn 实现词形还原"
    print('\n', '-' * 5, number_title, '-' * 5)

    text_train, y_train, text_test, y_test = load_data()

    import spacy, re
    regexp = re.compile('(?u)\\b\\w\\w+\\b')
    en_nlp = spacy.load('en_core_web_sm')
    old_tokenizer = en_nlp.tokenizer
    en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(regexp.findall(string))

    def custom_tokenizer(document):
        # 后面的两个参数取消了
        # doc_spacy = en_nlp(document, entity = False, parse = False)
        doc_spacy = en_nlp(document)
        return [token.lemma_ for token in doc_spacy]

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
    from sklearn.linear_model import LogisticRegression
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    # 仅使用1%的数据作为训练集来构建网格搜索，估计是为了尽快完成计算
    cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.99, train_size = 0.01, random_state = 0)
    classifier = LogisticRegression(solver = 'lbfgs', max_iter = 10000)

    # 利用标准的CountVectorizer进行网格搜索
    vectorizer = CountVectorizer(min_df = 5).fit(text_train)
    X_train = vectorizer.transform(text_train)
    X_test = vectorizer.transform(text_test)
    print('=' * 50)
    print("X_train.shape: {}".format(X_train.shape))
    grid_search_logistic_regression(X_train, y_train, X_test, y_test, classifier, param_grid, cv)
    # X_train.shape: (3000, 8285)，0.58，{'C': 10}，0.81
    # X_train.shape: (7000, 13809)，0.61，{'C': 1}，0.85
    # X_train.shape: (25000, 27271)
    # Best cross-validation score(standard CountVectorizer): 0.719
    # Best parameters(standard CountVectorizer): {'C': 1}

    # 利用词形还原技术进行网格搜索
    lemma_vectorizer = CountVectorizer(tokenizer = custom_tokenizer, min_df = 5).fit(text_train)
    X_train_lemma = lemma_vectorizer.transform(text_train)
    X_test_lemma = lemma_vectorizer.transform(text_test)
    print('=' * 50)
    print("X_train_lemma.shape: {}".format(X_train_lemma.shape))
    grid_search_logistic_regression(X_train_lemma, y_train, X_test_lemma, y_test, classifier, param_grid, cv)
    # X_train_lemma.shape: (3000, 6848)，0.58，{'C': 10}，0.81
    # X_train_lemma.shape: (7000, 11291)，0.62，{'C': 1}，
    # X_train_lemma.shape: (25000, 21692)
    # Best cross-validation score(lemmatization): 0.731
    # Best parameters(lemmatization): {'C': 1}
    pass


# 7.9 主题建模和文档聚类
def LDA_for_movie_reviews_with_ten_topics():
    number_title = "主题建模和文档聚类"
    print('\n', '-' * 5, number_title, '-' * 5)

    text_train, y_train, text_test, y_test = load_data()

    # 对于无监督的文本文档，通常会删除非常常见的单词，避免分析过程受到过大的影响。
    # 本例中删除出现频次不大于15%的单词，即删除在15%以上的文档中出现过的单词。
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features = 10000, max_df = .15)
    # vectorizer = CountVectorizer(max_features = 10000)
    X_train = vectorizer.fit_transform(text_train)
    print("X_train.shape=", X_train.shape)
    # X_train.shape = (15000, 10000)
    # n_topics变成n_components了
    from sklearn.decomposition import LatentDirichletAllocation
    # lda = LatentDirichletAllocation(n_topics = 10,learning_method = 'batch', max_iter = 25, random_state = 0,
    #                                 n_jobs = 3)
    lda = LatentDirichletAllocation(n_components = 10, learning_method = 'batch', max_iter = 25, random_state = 0,
                                    n_jobs = 3)
    documents_topics = lda.fit_transform(X_train)
    print("lda.components_.shape = ", lda.components_.shape)
    # lda.components_.shape = (10, 10000)

    sorting = np.argsort(lda.components_, axis = 1)[:, ::-1]
    feature_names = np.array(vectorizer.get_feature_names())
    mglearn.tools.print_topics(topics = range(10), feature_names = feature_names,
                               sorting = sorting, topics_per_chunk = 5, n_words = 10)

    # 取出第3个主题的文档内容，第3个主题的内容都与电视连续剧相关
    # topic 3：show, series, episode, tv, episodes, shows, season, new, television, years
    document = np.argsort(documents_topics[:, 3])[::-1]
    for i in document[:10]:
        print(b'.'.join(text_train[i].split(b'.')[:2]) + b'.\n')
        pass


def LDA_for_movie_reviews_with_hundred_topics():
    # 计算时间非常长，运行前注意！
    number_title = "主题建模和文档聚类应用于电影评论数据集"
    print('\n', '-' * 5, number_title, '-' * 5)

    text_train, y_train, text_test, y_test = load_data()

    # 对于无监督的文本文档，通常会删除非常常见的单词，避免分析过程受到过大的影响。
    # 本例中删除出现频次不大于15%的单词，即删除在15%以上的文档中出现过的单词。
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features = 10000, max_df = .15)
    # vectorizer = CountVectorizer(max_features = 10000)
    X_train = vectorizer.fit_transform(text_train)
    print("X_train.shape=", X_train.shape)

    # n_topics变成n_components了
    from sklearn.decomposition import LatentDirichletAllocation
    lda100 = LatentDirichletAllocation(n_components = 100, learning_method = 'batch', max_iter = 25, random_state = 0,
                                       n_jobs = 3)
    documents_topics100 = lda100.fit_transform(X_train)
    print("lda100.components_.shape = ", lda100.components_.shape)

    sorting = np.argsort(lda100.components_, axis = 1)[:, ::-1]
    feature_names = np.array(vectorizer.get_feature_names())
    topics = np.array([7, 16, 24, 25, 28, 36, 37, 45, 51, 53, 54, 63, 89, 97])
    mglearn.tools.print_topics(
            topics = topics, feature_names = feature_names,
            sorting = sorting, topics_per_chunk = 7, n_words = 20)

    # 按照第45个topic进行排序，第45个topic的内容都与音乐相关
    music = np.argsort(documents_topics100[:, 45])[::-1]
    for i in music[:10]:
        print(b".".join(text_train[i].split(b".")[:2]) + b".\n")
        pass

    fig, ax = plt.subplots(1, 2, figsize = (10, 10))
    topic_names = ["{:>2}".format(i) + " ".join(words) for i, words in enumerate(feature_names[sorting[:, :2]])]

    for col in [0, 1]:
        start = col * 50
        end = (col + 1) * 50
        ax[col].barh(np.arange(50), np.sum(documents_topics100, axis = 0)[start:end])
        ax[col].set_yticks(np.arange(50))
        ax[col].set_yticklabels(topic_names[start:end], ha = 'left', va = 'top')
        ax[col].invert_yaxis()
        # ax[col].set_xlim(0, 20)
        ax[col].set_xlim(0, 2000)
        yax = ax[col].get_yaxis()
        yax.set_tick_params(pad = 130)
        pass
    plt.suptitle("图7-6：LDA学到的主题权重")
    pass


# LDA似乎发现了两种主题：
#   - 特定类型的主题：与电影相关的评论
#   - 特定评分的主题：与评分相关的评论

if __name__ == "__main__":
    # 电影评论数据集
    # load_databases()

    # 7.3 将文本数据转换为数值表示（词袋）
    # transform_to_bag_of_words()

    # 7.3.2 将词袋应用于电影评论
    # bag_of_words_for_movie_reviews()

    # 删除那些不具有信息量的单词，即某些单词只出现在少量文档中，例如：数字、错误拼写的单词、生僻词等等
    # bag_of_words_for_movie_reviews_min_df()

    # 删除 不具有信息量的单词+停用词
    # bag_of_words_for_movie_reviews_stop_words()

    # 删除 不具有信息量的单词+停用词+出现在大量文档中的单词
    # bag_of_words_for_movie_reviews_max_df()

    # 7.5 用tf-idf缩放数据（提高计算的速度，不改变模型的精度）
    # bag_of_words_for_movie_reviews_tf_idf()

    # 7.7 多个单词的词袋（N元分词）
    # transform_to_n_gram()

    # 基于N元语法模型建立电影评论数据集的词袋
    # bag_of_words_for_movie_reviews_n_gram()

    # 7.8 词干提取和词形还原
    # word_normalization()

    # 了解正则表达式
    # test_regularization()

    # 通过使用CountVectorizer所使用的基于正则表达式的分词器来替换spacy的分词器
    # 计算时间非常长，运行前注意！
    # sklearn_lemmatization()

    # LDA_for_movie_reviews_with_ten_topics()

    # 主题建模和文档聚类应用于电影评论数据集
    # 计算时间非常长，运行前注意！
    LDA_for_movie_reviews_with_hundred_topics()
    import winsound

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
