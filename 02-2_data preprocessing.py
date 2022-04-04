##  데이터 전처리 필요성.  ##

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

# 2개 리스트 & 배열을 세로로 1개씩 붙이면 --> column_stack
# 2개 리스트 & 배열을 가로로 쭉 붙이면 --> concatenate
import numpy as np
fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))


## train_test_split()로 전달되는 리스트나 배열 나누기.
## train set / test set 구분
# np.random.seed(42)
# index = np.arange(49)
# np.random.shuffle(index)
from sklearn.model_selection import train_test_split        # 리스트나 배열을 비율에 따라 나누기.
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
print(train_input.shape, test_input.shape)
print(train_target.shape, test_target.shape)
print(test_target)
list_test_target = list(test_target)
print(list_test_target.count(1), list_test_target.count(0))
print()


## split 함수 & random_state로 무작위로 데이터를 나누었을 때, 샘플이 골고루 섞이지 않는다(샘플링 편향이 또 발생). ==> 해결: stratify
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
print(test_target)
list_test_target_stratify = list(test_target)
print(list_test_target_stratify.count(1), list_test_target_stratify.count(0))


## K-최근접 이웃(훈련 데이터를 저장하는 것으로만 훈련 진행)
from sklearn.neighbors import KNeighborsClassifier          # 어떤 규칙을 찾기보다는, 전체 데이터를 메모리에 가지고 있음.
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))
print("25, 150 인 경우 구분:", kn.predict([[25, 150]]))


## 수상한 도미 한마리 그리기
import matplotlib.pyplot as plt                             # matplotlib의 plot함수를 plt로 줄여서 사용.
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.scatter(25, 150, marker='^')


## investigation for K-최근접 이웃 데이터(이웃데이터 5개 참조)
distances, indexes = kn.kneighbors([[25, 150]])
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.title('investigation')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


print()
print("5개 sample --> length & weight:", train_input[indexes])
print("5개 sample --> 도미 & 빙어", train_target[indexes])
print("5개 sample <-> [25, 150] 거리", distances)


##------------------------------------------------------------------------------------
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlim((0, 1000))
plt.title("Re-scale")
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


## X축 범위가 좁고, y축은 넓어서 --> y축으로 조금만 멀어져도 거리가 아주 큰 값으로 계산.
## 이를 두 특성의 스케일(scale)이 다르다고 말한다.
## 특성값을 일정한 기준으로 맞춰주는 작업 ==> 데이터 전처리(Data preprocessing)
## 가장 널리 사용하는 전처리 방법 중 하나 ==> 표준점수(standard score & z점수)
## 표준점수는 각 특성값이 0에서 표준편차의 몇 배만큼 떨어져 있는지 확인.

mean = np.mean(train_input, axis=0)             # axis = 0은 세로를 의미 / axis = 1은 가로를 의미
std = np.std(train_input, axis=0)
print()
print("평균", mean, "표준편차", std)


## train_scaled 구하고 다시 훈련시키기
train_scaled = (train_input - mean) / std
kn.fit(train_scaled, train_target)


plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(25, 150, marker='^')
plt.title('[25, 150] not scaled')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


## [25, 150] 역시 훈련세트의 mean / std를 반영해야 함.
new = ([25, 150] - mean) / std
distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.title('final plot')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


## Test set 역시, mean & std 반영해야 함.
test_scaled = (test_input - mean) / std
print("평가: ", kn.score(test_scaled, test_target))
print("도미 vs 빙어 -->", kn.predict([new]))