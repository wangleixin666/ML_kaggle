# -*- coding: utf-8 -*

import pandas as pd
# 导入读取文件用的工具包
from sklearn.feature_extraction import DictVectorizer
# 导入用于特征向量化的工具包
import lightgbm as lgb
# 导入lightgbm工具包
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

dict_vec = DictVectorizer(sparse=False)

X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
# 对特征进行向量化处理
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

# print dict_vec.feature_names_
# 输出特征的向量化之后的结果

# print len(X_train)     # 891
# print len(X_test)      # 418

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 10,
    'verbose': 0,
    'metric': 'logloss',
    'max_bin': 255,
    'max_depth': 7,
    'learning_rate': 0.3,
    'nthread': 4,
    'n_estimators': 85,
    # 'feature_fraction': 0.8
    'feature_fraction': 1
}


def train_model(model_file='model/lgb'):

    lgb_train = lgb.Dataset(X_train, label=Y_train)
    lgb_eval = lgb.Dataset(X_test, label=Y_test, reference=lgb_train)

    print "begin train..."
    bst = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_eval],
        num_boost_round=160,
        early_stopping_rounds=10)
    print "train end\nsaving..."
    bst.save_model(model_file)
    return bst


def create_submission():
    # get model
    bst = train_model()

    # load test data
    test_df = pd.read_csv("data/test.csv", header=0)
    xg_test = test_df.iloc[:, :].values
    print "predicting..."
    pred = bst.predict(xg_test)
    print "predict end."
    # create csv file
    print "create submission file..."
    pred = map(lambda x: sum([i * round(y) for i, y in enumerate(x)]), pred)
    submission = pd.DataFrame({
        'ImageId': range(1, len(pred) + 1),
        'Label': [int(x) for x in pred]
    })
    # submission.to_csv("submission.csv", index=False)
    np.savetxt(
        'submission.csv',
        np.c_[range(1, len(pred) + 1), pred],
        delimiter=',',
        header='ImageId,Label',
        comments='',
        fmt='%d')
    print "----end----"


def tune_model():
    print "load data ..."
    dataset = pd.read_csv("data/train.csv", header=0)
    d_x = dataset.iloc[:, 1:].values
    d_y = dataset.iloc[:, 0].values

    print "create classifier..."
    param_grid = {
        # "reg_alpha": [0.3, 0.7, 0.9, 1.1],
        "learning_rate": [0.1, 0.25, 0.3],
        'n_estimators': [75, 80, 85, 90],
        'max_depth': [6, 7, 8, 9]
    }
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'max_bin': 255,
        'max_depth': 7,
        'learning_rate': 0.25,
        'n_estimators': 80,
    }
    # max_depth = 7, learning_rate:0.25
    model = lgb.LGBMClassifier(
        boosting_type='gbdt', objective="multiclass", nthread=8, seed=42)
    model.n_classes = 10
    print "run grid search..."
    searcher = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    searcher.fit(d_x, d_y)
    print searcher.grid_scores_
    print "=" * 30, '\n'
    print searcher.best_params_
    print "=" * 30, '\n'
    print searcher.best_score_
    print "end"


if __name__ == "__main__":
    # create_submission()
    tune_model()

"""
# specify your configurations as a dict
params = {
'task': 'train',
'boosting_type': 'gbdt',
'objective': 'binary',
'metric': {'l2', 'auc'},
'num_leaves': 31,
'learning_rate': 0.05,
'feature_fraction': 0.9,
'bagging_fraction': 0.8,
'bagging_freq': 5,
'verbose': 0
}

gbm = lgb.train(params,
lgb_train,
num_boost_round=20,
valid_sets=lgb_eval,
early_stopping_rounds=5)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print(y_pred)
print('The roc of prediction is:', roc_auc_score(y_test, y_pred) )
print('Dump model to JSON...')
# dump model to json (and save to file)
model_json = gbm.dump_model()
with open('lightgbm/model.json', 'w+') as f:
json.dump(model_json, f, indent=4)
print('Feature names:', gbm.feature_name())
print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))

"""



"""搜索最优参数配置
gs = GridSearchCV(xgbc_best, params, cv=5)
# 使用 n_jobs=-1 容易报错

gs.fit(X_train, Y_train)

print gs.best_score_
print gs.best_params_

xgbc_best_Y_predict = gs.predict(X_test)

xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_Y_predict})
xgbc_best_submission.to_csv('D://kaggle/titanic/datasets/xgbc_best_submission.csv', index=False)

"""

