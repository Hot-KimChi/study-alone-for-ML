##  데이터 전처리  ##
##  lenth / weight ==> 25 / 150

import matplotlib.pyplot as plt                             # matplotlib의 plot함수를 plt로 줄여서 사용.
from sklearn.neighbors import KNeighborsClassifier          # 어떤 규칙을 찾기보다는, 전체 데이터를 메모리에 가지고 있음.
import numpy as np
from sklearn.model_selection import train_test_split        # 리스트나 배열을 비율에 따라 나누기.


## Raw data: 도미와 빙어.
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

## 도미데이터와 빙어데이터 합치기
fish_length = bream_length + smelt_length
fish_weight = bream_weight + smelt_weight

## 넘파이의 column_stack() 함수는 전달받은 리스트를 일렬로 세운 다음 차례대로 나란히 연결.
## 넘파이의 np.ones() / np.zeros()로 Target데이터 생성 가능.
## 데이터가 클수록, 파이썬 리스트는 비효율적이므로 넘파이 배열사용(low level 언어)을 추천.
# fish_data = [[l, w] for l, w in zip(length, weight)]
# fish_target = [1] * 35 + [0] * 14
# input_arr = np.array(fish_data)
# target_arr = np.array(fish_target)

fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))


## train set / test set 구분
# np.random.seed(42)
# index = np.arange(49)
# np.random.shuffle(index)

## train_test_split()로 전달되는 리스트나 배열 나누기.
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
print(train_input.shape, test_input.shape)
print(train_target.shape, test_target.shape)
print(test_target)
list_test_target = list(test_target)
print(list_test_target.count(1), list_test_target.count(0))
print()


## split 함수 & random_state로 무작위로 데이터를 나누었을 때, 샘플이 골고루 섞이지 않는다. ==> stratify
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
print(test_target)
list_test_target_stratify = list(test_target)
print(list_test_target_stratify.count(1), list_test_target_stratify.count(0))


## K-최근접 이웃(훈련 데이터를 저장하는 것으로만 훈련 진행)
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))
print("25, 150 인 경우 구분:", kn.predict([[25, 150]]))


plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.scatter(25, 150, marker='^')


## investigation for K-최근접 이웃 데이터(이웃데이터 5개 참조)
distance, indexes = kn.kneighbors([[25, 150]])
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlabel('lenth')
plt.ylabel('weight')
plt.show()


print(train_input[indexes])
print(train_target[indexes])
print(distance)