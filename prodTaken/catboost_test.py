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

from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
# cbc = CatBoostClassifier(verbose = 200)

# parameters = param_cat = {"depth" : [11,12,13,15],
#           "iterations" : [1000, 1100, 1200, 1300],
#           "learning_rate" : [0.001, 0.05], 
#           "l2_leaf_reg" : [2],
#           "border_count" : [254]
#           }
# Grid_CBC = GridSearchCV(estimator=cbc, param_grid = parameters, cv = 5, n_jobs=-1)
# Grid_CBC.fit(x_train, y_train)

# print(Grid_CBC.best_params_)
# print(Grid_CBC.best_score_)

# cbc = CatBoostClassifier(border_count= 254, depth= 10, iterations= 1000, l2_leaf_reg= 2, learning_rate= 0.01, random_state=42)
cbc = CatBoostClassifier(border_count= 254, depth= 11, iterations= 1300, l2_leaf_reg= 2, learning_rate= 0.05, random_state=42)
cbc.fit(x_train,y_train)
val_predict = cbc.predict(x_val) 
from sklearn import metrics 
print('정확도 :', metrics.accuracy_score(y_val, val_predict))

# cbc.fit(x,y)
# pred = cbc.predict(test)
# print('----------------------예측된 데이터의 상위 10개의 값 확인--------------------\n')
# print(pred[:10])

# result = cbc.score(test, pred)
# ic('model.score:', result) 

# sample_submission['ProdTaken'] = pred
# ic(sample_submission.head())
# sample_submission.to_csv('submission/submission_catboost.csv',index = False)

'''
정확도 : 0.928388746803069
데이콘 : 0.92613
'''