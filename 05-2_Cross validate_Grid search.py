## 교차검증 이유: 이런저런 값으로 모델을 많이 만들어서 테스트 세트로 평가하면, 결국 테스트 세트에만 잘 맞는 모델이 된다.
## 테스트 세트를 사용하지 않으면, overfitting / underfitting 구분 불가능
## 테스트 세트를 사용하지 않고, 훈련세트를 또 나누어서 검증세트 생성해서 평가.

## 순서
## 1) 훈련세트 모델 훈련 & 검증세트로 모델 평가
## 2) 매개변수를 바꾸면서 가장 좋은 모델 선정.
## 3) 결정된 매개변수로 훈련 --> 전체 세트로 훈련(훈련세트 + 검증세트)
## 4) 테스트세트 최종점수 평가.


import pandas as pd
import numpy as np
wine = pd.read_csv('https://bit.ly/wine-date')


## data set / target set으로 나누기
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()


## train set / test set으로 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

## validation set / train set으로 나누기
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

print(sub_input.shape, val_input.shape)


## Decision Tree training
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)

print()
print(np.round_(dt.score(sub_input, sub_target), 3))
print(np.round_(dt.score(val_input, val_target), 3))


## cross validation(교차검증)
## cross validation 주의점 --> 훈련세트를 섞어 폴드를 나누지 않는다.
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print()
print(scores)
print('교차검증 R^2(5 fold):', np.round_(np.mean(scores['test_score']), 3))

## k-fold 구분 / 골고루 나누기 --> stratifiedkfold
from sklearn.model_selection import StratifiedKFold
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print('교차검증 R^2 & Stratifiedkfold(10 fold):', np.round_(np.mean(scores['test_score']), 3))


## Grid-search(그리드서치) --> hyper-parameter 튜닝.
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print()
print('DecisionTree R^2 & hyper-parameter by 그리드서치:', np.round_(dt.score(train_input, train_target), 3))
print(gs.best_estimator_)
print('0.0001 - 0.0005까지 교차검증 R^2', gs.cv_results_['mean_test_score'])

# 각 매개변수 중 최대 R^2 값 찾기
print()
print('검증')
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])

## 과정 정리
## 1) 탐색할 매개변수 지정
## 2) 훈련세트에서 그리드서치 수행 -> 최상의 R^2 점수값이 나오는 매개변수 조합을 찾는다
## 3) 전체 훈련세트(교차검증 때 사용한 훈련세트가 아니라)를 사용해 최종 모델 훈련

## 더 복잡한 매개변수 조합 진행
## 모델: 9 x 15 x 10 = 1350개 with 5 fold => 6750개
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print()
print('복잡한 매개변수(3개) 찾기:', gs.best_params_)
print('복잡한 매개변수로 setting R^2:', np.max(gs.cv_results_['mean_test_score']))
dt = gs.best_estimator_
print('최종 모델 결정 후, Test set R^2', dt.score(test_input, test_target))


## Random search(랜덤서치)
## 매개변수 값의 목록을 전달하는 것이 아니라 매개변수를 샘플링할 수 있는 확률 분포 객체 전달.
from scipy.stats import uniform, randint
# 싸이파이 - 균등하게 샘플링하는 법
rgen = randint(0, 10)
print(rgen.rvs(10))
print(np.unique(rgen.rvs(1000), return_counts=True))

ugen = uniform(0, 1)
print(ugen.rvs(5))
print(np.unique(ugen.rvs(5), return_counts=True))

# Random search parameter setting
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(5, 20),
          'min_samples_split': randint(2, 100),
          'min_samples_leaf': randint(1, 25)
          }
from sklearn.model_selection import RandomizedSearchCV
# 샘플링 횟수는 100번 setting.
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
print(gs.best_estimator_)
print(np.max(gs.cv_results_['mean_test_score']))
dt = gs.best_estimator_
print(dt.score(test_input, test_target))