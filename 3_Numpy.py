##  샘플 편향(sampling bias) 해결하기  ##

import matplotlib.pyplot as plt                             # matplotlib의 plot함수를 plt로 줄여서 사용.
from sklearn.neighbors import KNeighborsClassifier          # 어떤 규칙을 찾기보다는, 전체 데이터를 메모리에 가지고 있음.
import numpy as np


## Raw data: 도미와 빙어.
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


## 도미데이터와 빙어데이터 합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1] * 35 + [0] * 14


## 넘파이이용하여 2차원 변환 배열로 변경.
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

## 2차원 변환 배열을 랜덤하게 섞기. ==> index를 생성하여 index를 random으로 섞어서 index에 맞는 데이터 가져오기.
## 섞을 때, input_arr와 target_arr 같은 위치에 선택.
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)

# index에 맞는 데이터를 train set에 포함하기.
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# 잘 섞여서 생성되었는지 확인.
print(input_arr[13], train_input[0])

# train and test input 데이터 그리기.
# x축: 길이 / y축: 무게
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 훈련시키기
kn =KNeighborsClassifier()
kn.fit(train_input, train_target)
print("Neighbors_Accuracy = ", kn.score(test_input, test_target))

# Test 세트의 예측 결과와 실제 타깃확인.
print(kn.predict(test_input))
print(test_target)
