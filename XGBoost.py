# -*- coding: utf-8 -*

import pandas as pd
from sklearn.model_selection import train_test_split
# 用于数据分割
from sklearn.feature_extraction import DictVectorizer
# 用于数据向量化
from sklearn.ensemble import RandomForestClassifier
# 导入随机森林分类包
from xgboost import XGBClassifier
# 导入提升分类包

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# print titanic.info()

X = titanic[['pclass', 'age', 'sex']]
Y = titanic['survived']

X['age'].fillna(X['age'].mean(), inplace=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)

vec = DictVectorizer()
# 接下来进行特征向量化处理
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
print 'the accuracy of RandomForestClassifier is:', rfc.score(X_test, Y_test)

xgbc = XGBClassifier()
xgbc.fit(X_train, Y_train)
print 'the accuracy of XGBClassifier is:', xgbc.score(X_test, Y_test)