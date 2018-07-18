# -*- coding: utf-8 -*

from sklearn.feature_extraction.text import CountVectorizer
# 特征向量
import nltk
# 自然语言处理包

sent1 = 'The cat is walking in the bedroom.'
sent2 = 'A dog was running across the kitchen.'

count_vec = CountVectorizer()

sentences = [sent1, sent2]

# print count_vec.fit_transform(sentences).toarray()
# 输出特征向量化后的表示，以及各个维度的含义
# print count_vec.get_feature_names()

tokens_1 = nltk.word_tokenize(sent1)
# print tokens_1

tokens_2 = nltk.word_tokenize(sent2)
# print tokens_2

"""对句子进行词汇分割和正规化，I'm分割为I和'm"""

vocab_1 = sorted(set(tokens_1))
# 安装ASCII的排序输出
# print vocab_1

vocab_2 = sorted(set(tokens_2))
# print vocab_2

stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(t) for t in tokens_1]
# 初始化stemmer寻找各个词汇最原始的词根
# print stem_1

stem_2 = [stemmer.stem(t) for t in tokens_2]
# print stem_2

pos_tag_1 = nltk.tag.pos_tag(tokens_1)
# 初始化词性标注器，对每个词汇进行标注
print pos_tag_1

pos_tag_2 = nltk.tag.pos_tag(tokens_2)
print pos_tag_2