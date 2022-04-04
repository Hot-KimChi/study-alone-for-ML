## sklearn_linear model중, linearregression 이용하여 예측하기.
## polynomial regression(다항 회귀)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lr = LinearRegression()


perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])


train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
## sklearn 에 사용할 훈련(train) 세트는 꼭 2차원 배열 --> [[1],[2]] / 테스트(test) 세트는 1차원
train_input = np.reshape(train_input, (-1, 1))
test_input = np.reshape(test_input, (-1, 1))

# k-최근접 이웃 회귀 모델 훈련.
lr.fit(train_input, train_target)
print("50Cm 농어 무게 예측 with linear-regression:", lr.predict([[50]]))

# 머신러닝 알고리즘 모델파라미터 출력 ==> y = ax + b(a와 b를 출력)
print("y=ax+b 에서 a값:", lr.coef_, "/ y=ax+b 에서 b값:", lr.intercept_)

# 훈련 세트의 산점도
plt.scatter(train_input, train_target)
# 농어의 길이 15Cm --> 50Cm 까지 1차 방정식 그래프 그려보기(직선)
plt.plot([15, 50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_])

# 50Cm 농어 데이터 그리기
plt.scatter(50, lr.predict([[50]]), marker='^')
plt.title("linear plot")
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


## 훈련세트 R^2/ 테스트세트 R^2 확인
print()
print("훈련세트 R^2:", lr.score(train_input, train_target))
print("테스트세트 R^2:", lr.score(test_input, test_target))

## 훈련세트로 훈련하며, 보통 훈련세트의 R^2값이 더 높게 나온다.
## 훈련세트 R^2 >> 테스트세트 R^2  --> overfitting(실전 투입 시, 새로운 샘플에 대한 예측값이 잘 맞지 않는다.)
       ## - k 개수를 올려서, 여러 데이터를 반영.
## 훈련세트 R^2 << 테스트세트 R^2 또는 둘다 낮은 경우 --> underfitting(모델이 너무 단순해서 훈련세트에 적절히 훈련되지 않은 경우)
       ## - k 개수를 내려서 모델을 복잡하게 혹은 국지적인 패턴에 민감하게 변경
       ## - 훈련세트와 데이터의 세트의 크기가 작기 때문에 발생.
print("")
print("최적의 직선 ==> 최적의 곡선을 찾아라 y = ax^2 + bx + c ")

## 길이를 제곱한 항을 추가하여 train & fit ==> polynomial regression(다항 회귀)
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
print(train_poly.shape, test_poly.shape)

lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))
print(lr.coef_, lr.intercept_)


## 짧은 직선을 이어서 그리면 마치 곡선처럼 표현 가능(1씩 짧게 끊어서 그리기)
point = np.arange(15, 51)

plt.plot(point, lr.coef_[0] * point**2 + lr.coef_[1] * point + lr.intercept_)

plt.scatter(train_input, train_target)
plt.scatter(50, lr.predict([[50**2, 50]]), marker='^')
plt.title("curve plot")
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print()
print("훈련세트 R^2:", lr.score(train_poly, train_target))
print("테스트세트 R^2:", lr.score(test_poly, test_target))
## 테스트세트가 아직 점수가 높아, underfitting
       ## - 조금 더 복잡한 모델이 필요.