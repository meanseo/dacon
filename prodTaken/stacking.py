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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

ob_col = ['TypeofContact', 'Occupation',
            'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
train.replace('Fe Male', 'Female')

drop_col = ['NumberOfPersonVisiting','NumberOfChildrenVisiting', 'id', 'NumberOfTrips','NumberOfFollowups', 'OwnCar', 'MonthlyIncome'] 
train = train.drop(columns=drop_col)
test = test.drop(columns=drop_col)

# pandas의 fillna 메소드를 활용하여 NAN 값 채움
# 0 으로 채우는 경우
train.DurationOfPitch = train.DurationOfPitch.fillna(0)
test.DurationOfPitch = test.DurationOfPitch.fillna(0)

# mean 값으로 채우는 경우
mean_cols = ['Age', 'PreferredPropertyStar']
for col in mean_cols:
    train[col] = train[col].fillna(train[col].mean())
    test[col] = test[col].fillna(train[col].mean())

# "Unknown"으로 채우는 경우
train.TypeofContact = train.TypeofContact.fillna("Unknown")
test.TypeofContact = test.TypeofContact.fillna("Unknown")

encoder = LabelEncoder()
encoder.fit(train['TypeofContact'])
encoder.transform(train['TypeofContact'])

for col in ob_col:
    encoder = LabelEncoder()
    encoder.fit(train[col])
    train[col] = encoder.transform(train[col])
    test[col] = encoder.transform(test[col])

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(train[['Age', 'DurationOfPitch', 'PreferredPropertyStar']])

train[['Age', 'DurationOfPitch', 'PreferredPropertyStar']] = scaler.transform(train[['Age', 'DurationOfPitch', 'PreferredPropertyStar']])
test[['Age', 'DurationOfPitch', 'PreferredPropertyStar']] = scaler.transform(test[['Age', 'DurationOfPitch', 'PreferredPropertyStar']])

x = train.drop(columns=['ProdTaken'])
y = train['ProdTaken']

from sklearn.model_selection import train_test_split
from sklearn import model_selection
x_train,x_val,y_train,y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

rf = RandomForestClassifier(n_estimators=250, max_depth=15, random_state=42)
xgb = XGBClassifier(max_depth=14, n_estimators=250,
                    colsample_bylevel=0.8, 
                    colsample_bytree= 0.9, random_state=42)
knn_clf  = KNeighborsClassifier(metric= 'manhattan', n_neighbors= 6, weights= 'distance')
