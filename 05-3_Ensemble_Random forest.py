## Ensemble Learring: Random Forest
## 1) 안정적인 성능
## 2) 특징 2가지
##    - 훈련데이터에서 랜덤하게 샘플을 추출 --> 중복된 샘플 추출 가능: Bootstrap sample
##    -

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine-date')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print('Random Forest Train R^2:', np.round_(np.mean(scores['train_score']), 3))
print('Random Forest Train R^2:', np.round_(np.mean(scores['test_score']), 3))

