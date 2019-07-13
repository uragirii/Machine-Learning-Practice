import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Using the Boston housing dataset
dataset = pd.read_csv("Datasets/housing.csv")

X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]].values
y = dataset.iloc[:, 8].values

from sklearn.preprocessing import Imputer 
imp = Imputer()
X[:, 0:8] = imp.fit_transform(X[:, 0:8])

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:, 8] = lab.fit_transform(X[:, 8])

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features=[8])
X = one.fit_transform(X)
X = X.toarray()

# Feature Scling in this case doesn't improve the accuarcy in this case\
# So, we are not using StandardScaling in this.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Now training the model over the training set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

lin_reg.score(X_test, y_test)  # 63%
lin_reg.score(X_train, y_train)  # 65%
lin_reg.score(X, y)  # 64%
# So the model is not good enough for Linear Regression, or we may have to
# fine tune some things
