import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)


# by decreasing the test_size, we can increase the accuracy 
# When test_size = 0.1 then the accuracy reaches 68 %
# When test_size = 0.4 then the accuracy goes down to 58%

from sklearn.svm import SVC
svc = SVC()

svc.fit(X_train, y_train)

svc.score(X_test, y_test) # 61 % 
svc.score(X_train, y_train) # 100% 
# So the model is overfiting. 

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

log_reg.score(X_test, y_test)  # 93.7 %
log_reg.score(X_train, y_train)  # 96.7%

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

log_reg.score(X_test, y_test)  # 95.1%

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn.score(X_test, y_test)  # 94.4% 
