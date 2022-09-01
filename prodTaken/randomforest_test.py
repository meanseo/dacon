import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import seaborn as sns
from sklearn.model_selection import GridSearchCV
# from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import accuracy_score

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
from sklearn import model_selection
x_train,x_val,y_train,y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

# from sklearn.svm import SVC

model = RandomForestClassifier()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
xg_parameters ={'max_depth' : [2, 3, 5, 7, 10, 11, 14, 15] , 'n_estimators': [100, 150, 200, 250], 'random_state':[42]}
 
grid_search_xg = model_selection.GridSearchCV ( estimator = model,
                                               param_grid = xg_parameters,
                                               scoring = 'recall',
                                               cv = skf,
                                               n_jobs=-1 )
grid_search_xg.fit(x_train, y_train)
best_xg = grid_search_xg.best_estimator_
print(best_xg)
model.fit(x_train,y_train)
val_predict = model.predict(x_val) 
from sklearn import metrics 
print('정확도 :', metrics.accuracy_score(y_val, val_predict))
print(model.score(x_val,y_val))
print(model.score(x_val,val_predict))