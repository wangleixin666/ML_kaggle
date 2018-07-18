# -*- coding: utf-8 -*

import pandas as pd
# 导入读取文件用的工具包
from sklearn.feature_extraction import DictVectorizer
# 导入用于特征向量化的工具包
from sklearn.ensemble import RandomForestClassifier
# 导入随机森林工具包
from xgboost import XGBClassifier
# 导入XGBClassifier用于处理随机分类预测的问题
from sklearn.model_selection import cross_val_score
# 导入交叉验证工具包
from sklearn.model_selection import GridSearchCV
# 使用并行网格搜索的方式寻找更好的超参数组合，期待进一步提高XGBClassifier的性能

train = pd.read_csv('D://kaggle/titanic/datasets/train.csv')
test = pd.read_csv('D://kaggle/titanic/datasets/test.csv')

# print train.info()
# print test.info()
# 分别输出训练与测试数据的基本信息

selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']

X_train = train[selected_features]
X_test = test[selected_features]

Y_train = train['Survived']

# print X_train['Embarked'].value_counts()
# print X_test['Embarked'].value_counts()
# print X_test['Fare'].value_counts()
# 接下来对数据进行补充

X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)
# 用频率最高的‘S’来补充Embarked

X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
# 对于Age用平均值来进行补充

X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
# Fare意思是票价

# print X_train.info()
# print X_test.info()
# print X_test['Fare'].value_counts()

"""
出现警告
A value is trying to be set on a copy of a slice from a DataFrame
"""

dict_vec = DictVectorizer(sparse=False)

X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
# 对特征进行向量化处理
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

# print dict_vec.feature_names_
# 输出特征的向量化之后的结果

rfc = RandomForestClassifier()
# print cross_val_score(rfc, X_train, Y_train, cv=5).mean()
# 使用5折交叉验证的方法在训练集上对默认配置的RandomForestClassifier进行性能评估
# 并且获得平均分类准确性的得分

rfc.fit(X_train, Y_train)
rfc_Y_predict = rfc.predict(X_test)
# 使用默认配置的随机森林对结果进行预测

rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_Y_predict})
rfc_submission.to_csv('D://kaggle/titanic/datasets/rfc_submission.csv', index=False)
# 将对测试数据预测的结果存放在文件中

xgbc = XGBClassifier()

# print cross_val_score(xgbc, X_train, Y_train, cv=5).mean()
# 使用5折交叉验证的方法在训练集上对默认配置的XGBClassifier进行性能评估
# 并且获得平均分类准确性的得分

xgbc.fit(X_train, Y_train)
xgbc_Y_predict = xgbc.predict(X_test)
# 使用默认配置的XGBClassifier进行预测

xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_Y_predict})
xgbc_submission.to_csv('D://kaggle/titanic/datasets/xgbc_submission.csv', index=False)

params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}

xgbc_best = XGBClassifier()

gs = GridSearchCV(xgbc_best, params, cv=5)
# 使用 n_jobs=-1 容易报错

gs.fit(X_train, Y_train)

print gs.best_score_
print gs.best_params_

xgbc_best_Y_predict = gs.predict(X_test)

xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_Y_predict})
xgbc_best_submission.to_csv('D://kaggle/titanic/datasets/xgbc_best_submission.csv', index=False)