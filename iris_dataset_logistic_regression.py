import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# First load the iris dataset
from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Now using Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)


log_reg.score(X_test, y_test)  # 94.73%
log_reg.score(X_train, y_train)  # 96.42%
log_reg.score(X, y)  # 96%

y_pred = log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import precision_score, recall_score

precision_score(y_test, y_pred, average='micro')  # 94.73%
recall_score(y_test, y_pred, average='micro')  # 94.73%

