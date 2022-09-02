import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
import xgboost as xgb
import hyperopt


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

ob_col = ['TypeofContact', 'Occupation',
            'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
train.replace('Fe Male', 'Female')

# corr_df = train.corr()
# corr_df = corr_df.apply(lambda x: round(x ,2))
# pd.set_option('display.max_columns',None)
# # ic(corr_df)
# ax = sns.heatmap(corr_df, annot=True, annot_kws=dict(color='g'), cmap='Greys')
# # plt.savefig('corr.png')

drop_col = ['NumberOfPersonVisiting','NumberOfChildrenVisiting', 'id', 'NumberOfTrips','NumberOfFollowups', 'OwnCar', 'MonthlyIncome'] 
train = train.drop(columns=drop_col)
test = test.drop(columns=drop_col)

# pandas의 fillna 메소드를 활용하여 NAN 값을 채워니다.
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

# 결과를 확인합니다.
train.isna().sum()
test.isna().sum()


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

from sklearn.preprocessing import MinMaxScaler
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
x_train,x_val,y_train,y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

from hyperopt import hp

# max_depth는 5에서 20까지 1간격으로, min_child_weight는 1에서 2까지 1간격으로
# colsample_bytree는 0.5에서 1사이, learning_rate는 0.01에서 0.2사이 정규 분포된 값으로 검색. 
xgb_search_space = {
                    'max_depth': hp.quniform('max_depth', 1, 15, 1),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1)
               }
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from hyperopt import STATUS_OK

# fmin()에서 입력된 search_space값으로 입력된 모든 값은 실수형임. 
# XGBClassifier의 정수형 하이퍼 파라미터는 정수형 변환을 해줘야 함. 
# 정확도는 높은 수록 더 좋은 수치임. -1* 정확도를 곱해서 큰 정확도 값일 수록 최소가 되도록 변 h환
def objective_func(search_space):
    # 수행 시간 절약을 위해 n_estimators는 100으로 축소
    xgb_clf = XGBClassifier(
                            n_estimators=250, 
                            max_depth=int(search_space['max_depth']), 
                            colsample_bytree=search_space['colsample_bytree'], 
                            colsample_bylevel=search_space['colsample_bylevel'],
                            eval_metric='logloss')
    
    accuracy = cross_val_score(xgb_clf, x_train, y_train, scoring='accuracy', cv=5)
        
    # accuracy는 cv=3 개수만큼의 정확도 결과를 가지므로 이를 평균해서 반환하되 -1을 곱해줌. 
    return {'loss':-1 * np.mean(accuracy), 'status': STATUS_OK}

from hyperopt import fmin, tpe, Trials

trial_val = Trials()
best = fmin(fn=objective_func,
            space=xgb_search_space,
            algo=tpe.suggest,
            max_evals=50, # 최대 반복 횟수를 지정합니다.
            trials=trial_val, rstate=np.random.default_rng(seed=42))
print('best:', best)

print('colsample_bytree:{0}, colsample_bylevel{1}, max_depth{2}'.format(
                        round(best['colsample_bytree'], 5), round(best['colsample_bylevel'], 5)
                        ,int(best['max_depth'])
                        ))

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))

xgb_wrapper = XGBClassifier(n_estimators=250,
                            max_depth=int(best['max_depth']),
                            colsample_bytree=round(best['colsample_bytree'], 5),
                            colsample_bylevel=round(best['colsample_bylevel'], 5),
                           )

evals = [(x_train, y_train)]
xgb_wrapper.fit(x_train, y_train, early_stopping_rounds=50, eval_metric='logloss',
                eval_set=evals,
                verbose=True)

preds = xgb_wrapper.predict(x_val)
pred_proba = xgb_wrapper.predict_proba(x_val)[:, 1]

get_clf_eval(y_val, preds, pred_proba)




# print('정확도 :', accuracy_score(y_val, val_predict))
# print(model.score(x_val,y_val))
# print(model.score(x_val,val_predict))
# xgb_wrapper.fit(x,y)
# pred = xgb_wrapper.predict(test)
# print('----------------------예측된 데이터의 상위 10개의 값 확인--------------------\n')
# print(pred[:10])



# evals = [(x, y)]
# xgb_wrapper.fit(x, y, early_stopping_rounds=50, eval_metric='logloss',
#                 eval_set=evals,
#                 verbose=True)

# preds = xgb_wrapper.predict(test)
# pred_proba = xgb_wrapper.predict_proba(test)[:, 1]

# ic(pred_proba)
# # get_clf_eval(y_val, preds, pred_proba)

# # result = xgb_wrapper.score(test, preds)
# # ic('model.score:', result) 
# sample_submission['ProdTaken'] = preds
# ic(sample_submission.head())
# sample_submission.to_csv('submission/submission_wrapper.csv',index = False)