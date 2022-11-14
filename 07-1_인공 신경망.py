## 딥러닝과 인공 신경망 알고리즘을 이해하고 텐서플로 사용해 인공 신경망 모델
from tensorflow import keras
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()


## 10개의 데이터 이미지 보기.
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()


## 레이블 당 샘플 개수 확인.
print("train 크기 확인:", train_input.shape, train_target.shape)
print("test 크기 확인:", test_input.shape, test_target.shape)

import numpy as np

## 10개의 데이터의 Target
print([train_target[i] for i in range(10)])
print(np.unique(train_target, return_counts=True))


## SGDClassifier 모델.
## 60,000개의 데이터샘플이 너무 많기에, 샘플을 하나씩 꺼내서 모델을 훈련하는 방법이 효율적.
## 확률적 경사 하강법. --> 여러 특성 중 기울기가 가장 가파른 방향따라 이동(즉, 정규화 필수 진행)

## 픽셀은 0 - 255 사이의 정숫값을 가지고 있기에 255나누어 정규화 진행.
## SGDClassifier 모델은 1차원 데이터만 받아들이기에 1차원으로 변경 필요.
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

print("1차원으로 변경, 샘플 개수 확인:", train_scaled.shape)


## Cross_validate 진행.
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log', max_iter=5, random_state=42)
scores =cross_validate(sc, train_scaled, train_target, return_train_score=True, n_jobs=-1)
print('Train validation score with max_iter=5:', np.mean(scores['test_score']))
print('Train score with max_iter=5:', np.mean(scores['train_score']))


sc = SGDClassifier(loss='log', max_iter=9, random_state=42)
scores =cross_validate(sc, train_scaled, train_target, return_train_score=True, n_jobs=-1)
print('Train validation score with max_iter=9:', np.mean(scores['test_score']))
print('Train score with max_iter=9:', np.mean(scores['train_score']))


sc = SGDClassifier(loss='log', max_iter=20, random_state=42)
scores =cross_validate(sc, train_scaled, train_target, return_train_score=True, n_jobs=-1)
print('Train validation score with max_iter=20:', np.mean(scores['test_score']))
print('Train score with max_iter=20:', np.mean(scores['train_score']))

##-------------------------------------------

## 인공 신경망(ANN) 작업 시작.
## 인공 신경망에서는 교차 검증을 잘 사용하지 않고 검증 세트를 별도로 덜어내어 사용
## 1) 딥러닝 분야의 데이터셋은 충분히 크기 때문에, 검증 점수가 안정적
## 2) 교차 검증을 수행하기에 훈련 시간이 너무 오래 걸림.


## 기존에 rule대로 데이터를 나누어 훈련세트 / 검증세트로 구분
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

print("train_scaled 크기:", train_scaled.shape)
print("train_target 크기:", train_target.shape)


## dense layer만들기
dense = keras.layers.Dense(10, activation='softmax', input_shape=(28*28,))
    ## 1) 10: 뉴런 개수
    ## 2) activation = 'softmax' : 뉴런의 출력에 적용할 함수
    ## 3) input_shape: 입력의 크기


## 밀집층을 가진 신경망 모델 형성
model = keras.Sequential(dense)


## 케라스 모델을 훈련하기 전에 설정단계 진행해야 함.
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
## model 객체의 compile 메서드 수행 with 손실함수 종류 / 훈련과정에서 계산하고 싶은 측정값 설정.
## 다중 분류에서는 크로스 엔트로피 손실함수 사용
    # 1) 이진 분류: loss = 'binary_crossentropy'
    # 2) 다중 분류: loss = 'categorical_crossentropy'
## 타깃값을 해당 클래스만 1이고 나머지는 모두 0인 배열로 만드는 것을 원-핫 인코딩 --> sprase_categorical_crossentropy


## 모델 훈련 / 모델 평가
