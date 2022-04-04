## 지도학습 / 샘플링 편향 and 비지도학습
## 훈련세트 & 테스트세트 구분하여 훈련하기.

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


import matplotlib.pyplot as plt                             # matplotlib의 plot함수를 plt로 줄여서 사용.
plt.scatter(bream_length, bream_weight)                     # 리스트와 같은 형식을 표현.
plt.scatter(smelt_length, smelt_weight)                     # 리스트와 같은 형식을 표현.
plt.xlabel('length')                                        # x축: 길이
plt.ylabel('weight')                                        # y축: 무게
plt.show()


## 빙어와 도미의 리스트를 합쳐 한 개의 리스트로 표현
## fish data를 2차원 리스트로 만들기 --> length, weight
## 사이킷런은 머신러닝 패키지이며, 2차원 리스트가 필요
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length, weight)]
# 도미 1 / 빙어 0
fish_target = [1] * 35 + [0] * 14


train_input = fish_data[:35]                                # 슬라이싱으로 fish_data 중, 0-34개 훈련데이터
train_target = fish_target[:35]                             # 슬라이싱으로 fish_target 중, 0-34개 훈련데이터
test_input = fish_data[35:]                                 # 슬라이싱으로 fish_data 중, 35-마지막 테스트데이터
test_target = fish_target[35:]                              # 슬라이싱으로 fish_data 중, 35-마지막 테스트데이터


from sklearn.neighbors import KNeighborsClassifier          # 어떤 규칙을 찾기보다는, 전체 데이터를 메모리에 가지고 있음.
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)                           # 훈련시키기.
print("Neighbors_Accuracy = ", kn.score(test_input, test_target))
print("Accuracy 0인 이유", train_target, test_target)        # 샘플링 편향.


## 샘플링 편향을 해결하기 위해 아래와 같이 접근
## array 변경하여, 랜덤하게 샘플을 섞어서 train set / test set 나누기
import numpy as np
input_array = np.array(fish_data)
target_array = np.array(fish_target)
print(input_array.shape, target_array.shape)


## input_arr / target_arr은 같은 위치에서 선택되어야 한다.
## 해결하기 위해 인덱스를 만들어 인덱스를 랜덤하게 섞는다.
index = np.arange(49)                                       # arange
np.random.seed(42)                                          # 무작위 결과값을 책과 동일하게 하기 위해 지정.
np.random.shuffle(index)                                    # shuffle 주어진 배열을 무작위하게 섞는다.
print("잘 섞여있는 index:", index)


## 잘 섞여있는 index를 기준으로 train_set / test_set 나누기
train_input = input_array[index[:35]]
train_target = target_array[index[:35]]
test_input = input_array[index[35:]]
test_target = target_array[index[35:]]
print("검산:", train_input[0], input_array[13])

## 훈련시키기
## fit는 실행할 때마다 이전에 학습한 모든 것을 잃어버린다. 해결방안: kn = kn.fit(train_input, train_target)
kn.fit(train_input, train_target)
print(kn.score(train_input, train_target))
print(kn.score(test_input, test_target))
print(kn.predict(test_input))


## 데이터 그려보기
## 2차원 배열은 행과 열 인덱스를 콤마로 구분 / 모두 선택할 경우 모두 생략 가능
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()