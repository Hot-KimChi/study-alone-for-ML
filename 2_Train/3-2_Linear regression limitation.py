## Edge sample일 경우, estimation error --> 길이 50cm 일 경우, 예측.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

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

knr = KNeighborsRegressor(n_neighbors=3)

# k-최근접 이웃 회귀 모델 훈련.
knr.fit(train_input, train_target)
prediction = knr.predict([[50]])
print("50Cm 농어 무게 예측:", prediction)

distances, indexes = knr.kneighbors([[50]])

# 훈련세트 산점도.
plt.scatter(train_input, train_target)

# 훈련세트 중 이웃샘플 산점도
plt.scatter(train_input[indexes], train_target[indexes], marker='D')

# 50Cm 농어 데이터 산점도.
plt.scatter(50, prediction, marker='^')
plt.xlabel('length')
plt.ylabel('weight')


# 검산 investigation
print("50Cm 농어 무게 검산:", np.mean(train_target[indexes]))

# 추가 100Cm investigation
print("100Cm 농어 무게 예측", knr.predict([[100]]))
plt.scatter(100, knr.predict([[100]]), marker='^')
plt.show()

## k-최근접 이웃으로 regression은 한계점이 분명이 존재.
## 머신러닝 모델은 주기적으로 훈련을 진행 ==> 시간과 환경이 변화하면서 데이터도 변경되기에 주기적으로 새로운 훈련 데이터로 모델 훈련.