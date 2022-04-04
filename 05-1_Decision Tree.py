## 트리 알고리즘
## 전체 와인 데이터에서 화이트 와인 vs 레드 와인 구분
import numpy as np
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine-date')

## 처음 5개의 샘플 데이터 보기
print(wine.head())
## 데이터프레임의 데이터 타입과 누락 데이터 확인
print(wine.info())
## 열에 대한 간략한 통계 확인(최소, 최대, 평균)
print(wine.describe())


## 판다스 데이터프로임 데이터와 타겟 구분
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()


## 훈련세트와 테스트세트 구분
## test size는 20%로 설정(default: 25%, 샘플갯수가 충분히 많이 있기에)
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
print(train_input.shape, test_input.shape)


## 알코올 도수, 당도, pH 스케일이 달라 표준화 진행.
## 표준화 진행 순서: 클래스 불어오기 --> fit --> transform(train data / test data)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


## 표준점수로 변환된 train_scaled, test_scaled이용하여 로지스틱 회귀모델 훈련.
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print('로직스틱 회귀모델 train R^2:', np.round_(lr.score(train_scaled, train_target), 3))
print('로직스틱 회귀모델 test R^2:', np.round_(lr.score(test_scaled, test_target), 3))


## 설명하기 혹은 이유를 설명하기 쉬운 결정트리 적용
print()
print('DecisionTree 이용')
from sklearn.tree import DecisionTreeClassifier
## 책과 실습 결과물이 같게 하기위해서 random state set-up
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print('결정트리 train R^2:', np.round_(dt.score(train_scaled, train_target), 3))
print('결정트리 test R^2:', np.round_(dt.score(test_scaled, test_target), 3))


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 7))
## plot_tree(dt) --> 너무 복잡하여 이해하기가 힘들다.

## max_depth 1로 주면 루트노드만 표시, filled 클래스에 맞게 색칠, feature_name 특성 이름 전달 가능
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


## Test score가 overfitting되어 이를 해결하기 위한 방안
## 일반화가 되어, 가지치기 진행(branch: 자라날 수 있는 트리의 최대깊이 지정)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print('결정트리 train R^2 / max_depth = 3:', np.round_(dt.score(train_scaled, train_target), 3))
print('결정트리 test R^2 / max_depth = 3:', np.round_(dt.score(test_scaled, test_target), 3))
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


## 불순도(gini)를 가지고 샘플을 구분
## 특성의 scaling은 결정트리 알고리즘에 아무런 영향이 없다. 아래는 검증
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print('결정트리 train R^2 / max_depth = 3 & no scale:', np.round_(dt.score(train_input, train_target), 3))
print('결정트리 test R^2 / max_depth = 3 & no scale:', np.round(dt.score(test_input, test_target), 3))
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


## 결정트리에서 어떤 특성이 제일 중요한지 확인 가능
print(dt.feature_importances_)