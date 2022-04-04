## Linear regression 은 특성이 많을수록 엄청난 효과(R^2)를 발생.
## polynomialFeature 클래스 사용하여 편하게 훈련.
## feature engineering: 기존의 특성을 사용해 새로운 특성을 추출
    ## 특성끼리의 곱 e.g.) 길이 x 높이
    ## 특성끼리의 제곱.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://bit.ly/perch_csv")

## 넘파이 배열로 변경.
perch_full = df.to_numpy()
print(df)
print(perch_full)

## Target 데이터 준비
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

## perch_full / perch_weight를 훈련세트와 테스트 세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

## PolynomialFeatures 이용하기
## fit -> transform 가능.
poly = PolynomialFeatures()
poly.fit([[2, 3]])
print()
print(poly.transform([[2, 3]]))
print("출력 1이 추가된 이유: y = ax + bx + c * 1")

## 1을 제거하기 위해 PolynomialFeatures에서 include_bias=False
## train_input을 transform 진행.
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print("transform된 train 세트의 모양:", train_poly.shape)
print(poly.get_feature_names())

## test_input을 transform 진행.
test_poly = poly.transform(test_input)

## 다중 회귀 모델 훈련하기.
lr = LinearRegression()
lr.fit(train_poly, train_target)

## Train 세트 / test 세트 R^2의 확인
print("Train 세트 R^2:", lr.score(train_poly, train_target))
print("test 세트 R^2:", lr.score(test_poly, test_target))
print()

## 특성을 더 많이 추가할 경우, 3제곱 4제곱 ==> PolynomialFeatures degree 변경
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print("transform된 train 세트의 모양:", train_poly.shape)
print(poly.get_feature_names())

## test_input을 transform 진행.
test_poly = poly.transform(test_input)

## 다중 회귀 모델 훈련하기.
lr = LinearRegression()
lr.fit(train_poly, train_target)

## Train 세트 / test 세트 R^2의 확인
print("Train 세트 R^2:", lr.score(train_poly, train_target))
print("test 세트 R^2:", lr.score(test_poly, test_target))
## test 세트의 R^2 score가 음수이기에 특성을 줄여야 함
## 선형 회귀 모델의 경우 특성에 기울기의 크기를 작게 만드는 방법 사용.
## 특성의 스케일이 정규화되지 않으면 여기에 곱해지는 계수 값도 차이가 크게 발생.

##--> overfitting을 줄이는 다른 방법(regulation: 릿지 & 라쏘)