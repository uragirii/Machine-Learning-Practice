import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

X = X[y!=1]
y = y[y!=1]


# load LInear SVC

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)


from sklearn.svm import LinearSVC
lin_svc = LinearSVC()

lin_svc.fit(X_train, y_train)

lin_svc.score(X_test, y_test)
lin_svc.score(X_train, y_train)

y_pred = lin_svc.predict(X_train)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_pred)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)

svc.score(X_test, y_test)
svc.score(X_train, y_train)

