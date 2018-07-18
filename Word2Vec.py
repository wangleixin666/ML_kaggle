# -*- coding: utf-8 -*

from sklearn.datasets import fetch_20newsgroups
# 通过互联网即时下载数据
from bs4 import BeautifulSoup
# 导入BeautifulSoup，主要是从网页抓取数据
import nltk
import re
from gensim.models import word2vec
# 导入Word2Vec来观察词汇之间的关联度

news = fetch_20newsgroups(subset='all')

X, y = news.data, news.target

"""定义一个函数，将新闻中的每条语句剥离出来，并返回一个句子的列表"""
def news_to_sentences(news):
    news_text = BeautifulSoup(news, "html.parser").get_text()
    # BeautifulSoup(news)改成了BeautifulSoup(news, "html.parser")
    # 消除了警告
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zZ-z]]', '', sent.lower().strip()).split())
    return sentences

sentences = []

"""将长篇新闻剥离出来用来训练"""
for x in X:
    sentences += news_to_sentences(x)

num_features = 300
# 配置词向量的维度
min_word_count = 20
# 保证被考虑的词汇的频度
num_workers = 2
# 设定并行化训练使用CPU计算核心的数量
context = 5
downsampling = 1e-3
# 定义训练词向量的上下文窗口大小

model = word2vec.Word2Vec(sentences, workers=num_workers, \
                          size=num_features, min_count=min_word_count, \
                          window=context, sample=downsampling)
# 训练词向量模型

model.init_sims(replace=True)
# 这个设定代表当前的训练好的词向量为最终版，也可以加快模型的训练速度

print model.most_similar('morning')
# 找到与morning最相关的十个词汇

print model.most_similar('email')
# 与email相关的十个词汇