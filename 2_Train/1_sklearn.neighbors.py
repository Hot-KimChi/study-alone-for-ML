import matplotlib.pyplot as plt                         # matplotlib의 plot함수를 plt로 줄여서 사용.
from sklearn.neighbors import KNeighborsClassifier      # 어떤 규칙을 찾기보다는, 전체 데이터를 메모리에 가지고 있음.


bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


plt.scatter(bream_length, bream_weight)                 # 리스트와 같은 형식을 표현.
plt.scatter(smelt_length, smelt_weight)                 # 리스트와 같은 형식을 표현.
plt.xlabel('length')                                    # x축: 길이
plt.ylabel('weight')                                    # y축: 무게
plt.show()


length = bream_length + smelt_length                    # 두 리스트 합치기(도미 + 빙어)            ex) [l1, l2, l3....]
weight = bream_weight + smelt_weight                    # 두 리스트 합치기(도미 + 빙어)            ex) [w1, w2, w3....]
fish_data = [[l, w] for l, w in zip(length, weight)]    # 사이킷런 2차원 리스트만 활용가능 / zip 함수로 나열된 리스트 각각 하나씩 원소꺼내 반환.       ex) [[l1,w1], [l2, w2], .... ]
fish_target = [1] * 35 + [0] * 14                       # fish target 답안만들기: 1 --> 도미 / 0 --> 빙어
print("1) 사이킷런 패키지를 위한 2차원 리스트:", fish_data)


kn =KNeighborsClassifier()                              # k-최근접 알고리즘 클래스 객체 생성(Train X, 가장 가까운 데이터 참고: 기본값 n=5)
kn.fit(fish_data, fish_target)                          # 훈련시키기.
print("2) Neighbors_Accuracy = ", kn.score(fish_data, fish_target))         # 사이킷런 모델 평가하는 method
plt.scatter(30, 600, marker='^')
print("3) Test sample_구분 =", kn.predict([[30, 600]]))                     # 새로운 데이터를 받아 정답을 예측.


kn49 = KNeighborsClassifier(n_neighbors=49)                                 # 참고 데이터 49개로 모델.
kn49.fit(fish_data, fish_target)                                            # 훈련시키기
print("4) n=49 data_Accuracy = ", kn49.score(fish_data, fish_target))       # 모델 평가
print("5) 35/49", 35/49)


print("")
print("6) <100% 정확도 --> n개 찾기>")
kn_opt = KNeighborsClassifier()
kn_opt.fit(fish_data, fish_target)

for n in range(5, 50):
    kn_opt.n_neighbors = n                              # 최근접 이웃 개수 설정
    score = kn_opt.score(fish_data, fish_target)        # 점수 계산
    if score < 1:                                       # 100% 정확도에 미치지 못하는 이웃 개수 출력
        print("최근접 이웃 = ", n, "/", "Accuracy = ", score)
        break