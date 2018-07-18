# -*- coding: utf-8 -*

import pandas as pd
# 用于读取数据
from bs4 import BeautifulSoup
# 用于整洁原始文本
import re
# 导入正则表达式工具包
from nltk.corpus import stopwords
# 导入停用词列表
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# 导入文本特征抽取器
from sklearn.naive_bayes import MultinomialNB
# 导入朴素贝叶斯模型
from sklearn.pipeline import Pipeline
# 用于方便搭建系统流程
from sklearn.model_selection import GridSearchCV
# 用于超参数组合的网格搜索
import nltk.data
from gensim.models import word2vec
# 导入word2vec工具包
from gensim.models import Word2Vec
# 直接导入训练好的词向量模型
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
# 进行影评情感分析


train = pd.read_csv('D:\kaggle\Bag of Words\datasets\labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('D:\kaggle\Bag of Words\datasets\\testData.tsv', delimiter='\t')
# 因为\t是python的关键词，所以加上一个“\”转义作用

# print train.head()
# print test.head()

# 定义函数，完成对原始评论的三项数据预处理任务
def review_to_text(review, remove_stopwords):
    raw_text = BeautifulSoup(review, 'html.parser').get_text()
    # 任务一、去除html标记
    letters = re.sub('[^a-zA-Z]', ' ', raw_text)

    """字母和空格！！！所以要把空格留出来，否则结果不准确"""

    words = letters.lower().split()
    # 变成小写字母
    # 任务二、去除非字母字符
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    # 任务三、如果remove_stopwords被激活，则去除评论中的停用词
    return words
    # 返回经过这三项预处理任务的词汇列表

X_train = []
for review in train['review']:
    X_train.append(' '.join(review_to_text(review, True)))
X_test = []
for review in test['review']:
    X_test.append(' '.join(review_to_text(review, True)))
# 分别对训练数据和测试数据进行上述三项预处理任务
# 其中的空格！！不能省略
Y_train = train['sentiment']

# print X_train[0]
# print X_test[0]

"""分别使用两种特征抽取器用Pipeline搭建两个朴素贝叶斯模型分类器"""

pip_count = Pipeline([('count_vec', CountVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer='word')), ('mnb', MultinomialNB())])

params_count = {'count_vec__binary': [True, False], 'count_vec__ngram_range': [(1, 1), (1, 2)], 'mnb__alpha': [0.1, 1.0, 10.0]}
params_tfidf = {'tfidf_vec__binary': [True, False], 'tfidf_vec__ngram_range': [(1, 1), (1, 2)], 'mnb__alpha': [0.1, 1.0, 10.0]}
# 分别配置用于模型超参数搜索的组合

"""因此需要将params_count中的count_vect_binary、count_vect_ngram_range、mnb_alpha改为count_vect__binary、count_vect__ngram_range、mnb__alpha"""

# 并行搜索
gs_count = GridSearchCV(pip_count, params_count, cv=4, verbose=1)
gs_count.fit(X_train, Y_train)
# 使用4折交叉验证对CountVectorizer的朴素贝叶斯模型进行并行化超参数搜索
# 单线程网格搜索

print gs_count.best_score_
print gs_count.best_params_
# 输出交叉验证中最佳的准确性得分和超参数组合

count_Y_predict = gs_count.predict(X_test)

# 同样的方法对TfidVectorizer的朴素贝叶斯模型进行并行化超参数搜索
gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv=4, verbose=1)
gs_tfidf.fit(X_train, Y_train)
# 使用4折交叉验证对CountVectorizer的朴素贝叶斯模型进行并行化超参数搜索
# 单线程网格搜索去掉n_jobs=-1

print gs_tfidf.best_score_
print gs_tfidf.best_params_
# 输出交叉验证中最佳的准确性得分和超参数组合

tfidf_Y_predict = gs_tfidf.predict(X_test)

submission_count = pd.DataFrame({'id': test['id'], 'sentiment': count_Y_predict})
submission_tfidf = pd.DataFrame({'id': test['id'], 'sentiment': tfidf_Y_predict})
# 使用pandas对需要提交的数据进行格式化

# submission_count.to_csv('D:\kaggle\Bag of Words\datasets\submission_count.csv', index=False)
# submission_tfidf.to_csv('D:\kaggle\Bag of Words\datasets\submission_tfidf.csv', index=False)
# 结果输出到硬盘保存

unlabeled_train = pd.read_csv('D:\kaggle\Bag of Words\datasets\unlabeledTrainData.tsv', delimiter='\t', quoting=3)
# 读取未标记的数据

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# 准备使用nltk的tokenizer对影评中的英文句子进行分割

def review_to_sentences(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_text(raw_sentence, False))
        return sentences
# 定义函数逐条对语句进行分句

corpora = []

for review in unlabeled_train['review']:
    corpora += review_to_sentences(review.decode('utf8'), tokenizer)
# 准备用于训练词向量的数据

num_features = 300
min_word_count = 20
num_workers = 4
context = 10
downsampling = 1e-3
# 配置训练词向量模型的超参数

model = word2vec.Word2Vec(corpora, workers=num_workers, \
                          size=num_features, min_count=min_word_count, \
                          window=context, sample=downsampling)

# 开始词向量模型的训练

model.init_sims(replace=True)

model_name = "D:\kaggle\Bag of Words\datasets\\300features_20minwords_10context"
# 将词向量模型的训练结果长期保存于电脑硬盘
model.save(model_name)

model = Word2Vec.load("D:\kaggle\Bag of Words\datasets\\300features_20minwords_10context")
# model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# print model.most_similar("man")
# 检查训练好的词向量模型效果

# print model.doesnt_match("man woman child kitchen".split())

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec
# 定义一个函数使用词向量产生文本特征向量

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVecs
# 定义另一个每条影评转化为基于词向量的特征向量

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_text(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

clean_test_reviews = []
for review in test["review"]:
    clean_train_reviews.append(review_to_text(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
# 准备新的基于词向量表示的训练和测试特征向量

gbc = GradientBoostingClassifier()

params_gbc = {'n_estimators': [10, 100, 500], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [2, 3, 4]}
# 配置超参数搜索组合

gs = GridSearchCV(gbc, params_gbc, cv=4, verbose=1)

gs.fit(trainDataVecs, Y_train)

print gs.best_score_
print gs.best_params_
# 输出最佳性能和最优超参数组合

result = gs.predict(testDataVecs)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
# 使用超参数调优之后的梯度上升树模型进行预测
output.to_csv("D:\kaggle\Bag of Words\datasets\submission_w2v.csv", index=False, quoting=3)