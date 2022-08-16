import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

'''
id : 샘플 아이디
Age : 나이
TypeofContact : 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색)
CityTier : 주거 중인 도시의 등급. (인구, 시설, 생활 수준 기준) (1등급 > 2등급 > 3등급) 
DurationOfPitch : 영업 사원이 고객에게 제공하는 프레젠테이션 기간
Occupation : 직업
Gender : 성별
NumberOfPersonVisiting : 고객과 함께 여행을 계획 중인 총 인원
NumberOfFollowups : 영업 사원의 프레젠테이션 후 이루어진 후속 조치 수
ProductPitched : 영업 사원이 제시한 상품
PreferredPropertyStar : 선호 호텔 숙박업소 등급
MaritalStatus : 결혼여부
NumberOfTrips : 평균 연간 여행 횟수
Passport : 여권 보유 여부 (0: 없음, 1: 있음)
PitchSatisfactionScore : 영업 사원의 프레젠테이션 만족도
OwnCar : 자동차 보유 여부 (0: 없음, 1: 있음)
NumberOfChildrenVisiting : 함께 여행을 계획 중인 5세 미만의 어린이 수
Designation : (직업의) 직급
MonthlyIncome : 월 급여
ProdTaken : 여행 패키지 신청 여부 (0: 신청 안 함, 1: 신청함)
'''
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

# [1955 rows x 20 columns]
# non: Age, TypeofContact, DurationOfPitch, NumberOfFollowups, PreferredPropertyStar,
# NumberOfTrips, NumberOfChildrenVisiting, MonthlyIncome
'''
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   id                        1955 non-null   int64
 1   Age                       1861 non-null   float64
 2   TypeofContact             1945 non-null   object
 3   CityTier                  1955 non-null   int64
 4   DurationOfPitch           1853 non-null   float64
 5   Occupation                1955 non-null   object
 6   Gender                    1955 non-null   object
 7   NumberOfPersonVisiting    1955 non-null   int64
 8   NumberOfFollowups         1942 non-null   float64
 9   ProductPitched            1955 non-null   object
 10  PreferredPropertyStar     1945 non-null   float64
 11  MaritalStatus             1955 non-null   object
 12  NumberOfTrips             1898 non-null   float64
 13  Passport                  1955 non-null   int64
 14  PitchSatisfactionScore    1955 non-null   int64
 15  OwnCar                    1955 non-null   int64
 16  NumberOfChildrenVisiting  1928 non-null   float64
 17  Designation               1955 non-null   object
 18  MonthlyIncome             1855 non-null   float64
 19  ProdTaken                 1955 non-null   int64
'''
# ic(train.info())
# ic(plt.hist(train.ProdTaken))
# ic(plt.show())
# ic(train.isna().sum())
# num_col = ['Age', 'DurationOfPitch']
ob_col = ['TypeofContact', 'Occupation',
          'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
ic(train['Gender'])

train.replace('Fe Male', 'Female')
ic(train['Gender'])

for col, dtype in train.dtypes.items():
    if dtype == 'object':
        # 문자형 칼럼의 경우 'Unknown'을 채워줍니다.
        value = 'Unknown'
        train.loc[:,col] = train[col].fillna(value)
        test.loc[:,col] = test[col].fillna(value)
    elif dtype == int or dtype == float:
        # 수치형 칼럼의 경우 0을 채워줍니다.
        value = 0
        train.loc[:,col] = train[col].fillna(value)
        test.loc[:,col] = test[col].fillna(value)
# print(train.isna().sum())

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

corr_df = train.corr()
corr_df = corr_df.apply(lambda x: round(x ,2))
pd.set_option('display.max_columns',None)
# ic(corr_df)
ax = sns.heatmap(corr_df, annot=True, annot_kws=dict(color='g'), cmap='Greys')
# plt.savefig('corr.png')

model = RandomForestClassifier()
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

x_train = train.drop(columns=['ProdTaken'])
y_train = train[['ProdTaken']]

model.fit(x_train,y_train)
ic(model.score(x_train, y_train).round(3))

# prediction = model.predict(test)
# print('----------------------예측된 데이터의 상위 10개의 값 확인--------------------\n')
# print(prediction[:10])

# result = model.score(test, prediction)
# ic('model.score:', result) 

# sample_submission['ProdTaken'] = prediction
# ic(sample_submission.head())
# sample_submission.to_csv('submission.csv',index = False)