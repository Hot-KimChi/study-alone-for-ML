## 1) length가지고 weight 예측하기
## 2) R^2구하기
## 3) 실제 모델에서(test)에서 error 확인(gram으로)
## 4) underfittng / overfitting 확인하기
## 5) 길이가 5 - 45일 경우, n값이 커짐에 따라 모델이 단순해 지는지 확인.(n 개수를 크게 하여 전반적인 영향을 끼치게)

import numpy as np
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


from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)


## sklearn 에 사용할 훈련(train) 세트는 꼭 2차원 배열 --> [[1],[2]] / 테스트(test) 세트는 1차원
train_input = np.reshape(train_input, (-1, 1))
test_input = np.reshape(test_input, (-1, 1))
print("train set 배열 크기:", train_input.shape, "test set 배열 크기:", test_input.shape)
print()


## KNeighborsRegressor 훈련시키기
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print("test R^2 with default(5):", knr.score(test_input, test_target))


## 실제 타깃과 예측한 값 사이의 차이 구하기
from sklearn.metrics import mean_absolute_error
test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print("|타깃 - 예측|:", mae)


## train 데이터 R^2 값 구해보기 --> 일반적으로 모델을 훈련 세트에서 훈련하면 훈련세트에 잘 맞는 모델 만들어 진다.
print("train R^2 with default(5):", knr.score(train_input, train_target))
print()


## 이웃의 개수를 3개로 지정.
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print("test R^2 with n = 3:", knr.score(test_input, test_target))
print("train R^2 with n = 3:", knr.score(train_input, train_target))


## 길이가 5 - 45까지 n이 커짐에 따라 모델이 단순해 지는지 확인.
x = np.arange(5, 45).reshape(-1, 1)

for n in [1, 5, 10]:
    knr.n_neighbors = n
    knr.fit(train_input, train_target)
    prediction = knr.predict(x)

    import matplotlib.pyplot as plt
    plt.scatter(train_input, train_target)
    plt.plot(x, prediction)
    plt.title("n_neights = {}".format(n))
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()