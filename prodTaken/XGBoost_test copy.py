import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier 
import hyperopt
from sklearn.model_selection import KFold, StratifiedKFold

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

ob_col = ['TypeofContact', 'Occupation',
            'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
train.replace('Fe Male', 'Female')

# 상관계수
# corr_df = train.corr()
# corr_df = corr_df.apply(lambda x: round(x ,2))
# pd.set_option('display.max_columns',None)
# # ic(corr_df)
# ax = sns.heatmap(corr_df, annot=True, annot_kws=dict(color='g'), cmap='Greys')
# # plt.savefig('corr.png')

drop_col = ['NumberOfPersonVisiting','NumberOfChildrenVisiting', 'id', 'NumberOfTrips','NumberOfFollowups', 'OwnCar', 'MonthlyIncome'] 
train = train.drop(columns=drop_col)
test = test.drop(columns=drop_col)

# pandas의 fillna 메소드를 활용하여 NAN 값 채움
# 0 으로 채우는 경우
train.DurationOfPitch = train.DurationOfPitch.fillna(0)
test.DurationOfPitch = test.DurationOfPitch.fillna(0)
# train.NumberOfFollowups = train.NumberOfFollowups.fillna(0)
# test.NumberOfFollowups = test.NumberOfFollowups.fillna(0)

# train.MonthlyIncome = train.MonthlyIncome.fillna(0)
# test.MonthlyIncome = test.MonthlyIncome.fillna(0)

# mean 값으로 채우는 경우
mean_cols = ['Age', 'PreferredPropertyStar']
for col in mean_cols:
    train[col] = train[col].fillna(train[col].mean())
    test[col] = test[col].fillna(train[col].mean())

# "Unknown"으로 채우는 경우
train.TypeofContact = train.TypeofContact.fillna("Unknown")
test.TypeofContact = test.TypeofContact.fillna("Unknown")

# 결과를 확인
# print(train.isna().sum())
# print(test.isna().sum())


encoder = LabelEncoder()
encoder.fit(train['TypeofContact'])
encoder.transform(train['TypeofContact'])

for col in ob_col:
    encoder = LabelEncoder()
    encoder.fit(train[col])
    train[col] = encoder.transform(train[col])
    test[col] = encoder.transform(test[col])
# ic(train)
# ic(test)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(train[['Age', 'DurationOfPitch', 'PreferredPropertyStar']])

# train[['Age', 'DurationOfPitch']] = scaler.transform(train[['Age', 'DurationOfPitch']])
# test[['Age', 'DurationOfPitch']] = scaler.transform(test[['Age', 'DurationOfPitch']])

train[['Age', 'DurationOfPitch', 'PreferredPropertyStar']] = scaler.transform(train[['Age', 'DurationOfPitch', 'PreferredPropertyStar']])
test[['Age', 'DurationOfPitch', 'PreferredPropertyStar']] = scaler.transform(test[['Age', 'DurationOfPitch', 'PreferredPropertyStar']])
# ic(train)
# test

x = train.drop(columns=['ProdTaken'])
y = train['ProdTaken']

from sklearn.model_selection import train_test_split
from sklearn import model_selection
x_train,x_val,y_train,y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

# model = XGBClassifier(max_depth=1, n_estimators=250,
#                     colsample_bylevel=0.8135699040313069, 
#                     colsample_bytree=0.9815958451244695, random_state=42
# )
# best: {'colsample_bylevel': 0.8135699040313069, 'colsample_bytree': 0.9815958451244695, 'learning_rate': 0.08992451546482534, 'min_child_weight': 1.0}

# model = XGBClassifier(random_state=42, colsample_bylevel=0.9, colsample_bytree=0.8)
# xg_parameters ={'max_depth' : [2, 3, 5, 7, 10, 11, 14, 15] , 'n_estimators': [100, 150, 200, 250, 300]}
 
# grid_search_xg = model_selection.GridSearchCV ( estimator = model,
#                                                param_grid = xg_parameters,
#                                                scoring = 'recall',
#                                                cv = 5,
#                                                n_jobs=-1 )
# grid_search_xg.fit(x_train, y_train)
# best_xg = grid_search_xg.best_estimator_
# print(best_xg)

# best_xg.fit(x_train, y_train)
# preds = best_xg.predict(x_val)
# # pred_proba = best_xg.predict_proba(x_val)[:, 1]
# ic('score:', accuracy_score(preds, y_val))

# colsample_bytree:0.85127, learning_rate:0.06776



from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score

# # 초모수 탐색공간 정의
# param_space = {
#                 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
#                 'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1)
#                }

# # 목적함수 정의
# def objective(params):
#     params = {
#               'colsample_bytree' : round(params['colsample_bytree'], 5),
#               'colsample_bylevel' : round(params['colsample_bylevel'], 5)
#               }
#     xgb_clf = XGBClassifier(n_estimators=250, max_depth=10, **params) 
#     best_score = cross_val_score(xgb_clf, x_train, y_train, 
#                                  scoring='accuracy', 
#                                  cv=5, 
#                                  n_jobs=-1).mean()
#     loss = 1 - best_score
#     return loss

# # 알고리즘 실행
# best = fmin(fn=objective, space=param_space, 
#             max_evals=20, 
#             algo=tpe.suggest)
# print(best)

model = XGBClassifier(max_depth=14, n_estimators=250,
                    colsample_bylevel=0.8, 
                    colsample_bytree= 0.9,
)

model.fit(x_train, y_train)
preds = model.predict(x_val)
# pred_proba = best_xg.predict_proba(x_val)[:, 1]
ic('score:', accuracy_score(preds, y_val))

# model.fit(x,y)
# pred = model.predict(test)
# print('----------------------예측된 데이터의 상위 10개의 값 확인--------------------\n')
# print(pred[:10])

# result = model.score(test, pred)
# ic('model.score:', result) 
# sample_submission['ProdTaken'] = pred
# ic(sample_submission.head())
# sample_submission.to_csv('submission/submission_xgboost_param2.csv',index = False)