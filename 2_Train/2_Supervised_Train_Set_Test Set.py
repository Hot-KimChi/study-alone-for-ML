##  지도학습 / 샘플링 편향 and 비지도학습  ##

import matplotlib.pyplot as plt                             # matplotlib의 plot함수를 plt로 줄여서 사용.
from sklearn.neighbors import KNeighborsClassifier          # 어떤 규칙을 찾기보다는, 전체 데이터를 메모리에 가지고 있음.
import numpy as np

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


plt.scatter(bream_length, bream_weight)                     # 리스트와 같은 형식을 표현.
plt.scatter(smelt_length, smelt_weight)                     # 리스트와 같은 형식을 표현.
plt.xlabel('length')                                        # x축: 길이
plt.ylabel('weight')                                        # y축: 무게
plt.show()


length = bream_length + smelt_length
weight = bream_weight + smelt_weight
fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1] * 35 + [0] * 14


train_input = fish_data[:35]                                # 슬라이싱으로 fish_data 중, 0-34개 훈련데이터
train_target = fish_target[:35]                             # 슬라이싱으로 fish_target 중, 0-34개 훈련데이터
test_input = fish_data[35:]                                 # 슬라이싱으로 fish_data 중, 35-마지막 테스트데이터
test_target = fish_target[35:]                              # 슬라이싱으로 fish_data 중, 35-마지막 테스트데이터


kn =KNeighborsClassifier()
kn.fit(train_input, train_target)                           # 훈련시키기.
print("Neighbors_Accuracy = ", kn.score(test_input, test_target))
print("Accuracy 0인 이유", train_target, test_target)        # 샘플링 편향.