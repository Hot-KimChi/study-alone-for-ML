        ## 인공신경망에 층을 여러 개 추가하여 케라스로 심층 신경망을 만든다.

from tensorflow import keras
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()

        ## 정규화 / reshape / 훈련세트 & 검증세트 세분화.
from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)


        ## 인공신경망 모델에 층을 2개 추가.(hidden layer)
        ## 1) 은닉층의 활성화 함수는 비교적 자유롭게 선택.
        ## 2) dense 제일 앞 층은 input_shape 매개변수 지정.
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')

        ## 심층 신경망 모델 만들기.
        ## 주의: 출력층을 가장 마지막에 두어야 한다 / 리스트는 가장 처음 등장하는 은닉층에서 마지막 출력층 순서로 나열.
model = keras.Sequential([dense1, dense2])