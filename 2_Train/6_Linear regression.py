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

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
## sklearn 에 사용할 훈련/테스트 input 세트는 꼭 2차원 배열 --> [[1],[2]] / 타깃 세트는 1차원
train_input = np.reshape(train_input, (-1, 1))
test_input = np.reshape(test_input, (-1, 1))
print("train set 배열 크기:", train_input.shape, "test set 배열 크기:", test_input.shape)
print()

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print("R^2 = 1 - { sum[(타깃 - 예측)^2] / sum[(타깃 - 평균)^2 }")
print("test set R^2(결정계수):", knr.score(test_input, test_target))
print("train set R^2(결정계수):",knr.score(train_input, train_target))

## 훈련세트로 훈련하며, 보통 훈련세트의 R^2값이 더 높게 나온다.
## 훈련세트 R^2 >> 테스트세트 R^2  --> overfitting(실전 투입 시, 새로운 샘플에 대한 예측값이 잘 맞지 않는다.)
## - k 개수를 올려서, 여러 데이터를 반영.
## 훈련세트 R^2 << 테스트세트 R^2 또는 둘다 낮은 경우 --> underfitting(모델이 너무 단순해서 훈련세트에 적절히 훈련되지 않은 경우)
## - k 개수를 내려서 모델을 복잡하게 혹은 국지적인 패턴에 민감하게 변경
## - 훈련세트와 데이터의 세트의 크기가 작기 때문에 발생.

## 테스트 세트에 대한 예측값 / 평균 절대값 오차 계산.
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)
print()
print("예측이 타깃대비 평균적으로 error 수치:", mean_absolute_error(test_target, test_prediction))
print()

knr.n_neighbors = 3
knr.fit(train_input, train_target)
print("k=3 test set R^2(결정계수):", knr.score(test_input, test_target))
print("k=3 train set R^2(결정계수):", knr.score(train_input, train_target))