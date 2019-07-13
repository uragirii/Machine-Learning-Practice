import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# This dataset has been taken from UCI ML Repository
# Site : https://archive.ics.uci.edu/ml/datasets/Air+Quality
# The csv file was not appropriate(?) or correct, so reading from xlsx file
datasets = pd.read_excel("Datasets/AirQualityUCI.xlsx")

# The first and Second column are Date and Time, till now idk date and time
# preprocessing so, I will crop the dataset, also, here we will have to train 3
# different models, for Temp, Relative Humidity and absolute Humidity
X = datasets.iloc[:, 2:-3].values
y_temp = datasets.iloc[:, -3].values
y_relHumid = datasets.iloc[:, -2].values
y_absHumid = datasets.iloc[:, -1].values


# Now, I've not checked whether the data have a NaN value or not.
# First divinding the data into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_temp)


# Traing the model using Linear Regression
from sklearn.linear_model import LinearRegression
lin_alg = LinearRegression()
lin_alg.fit(X_train, y_train)

# Testing the accuracy by score
lin_alg.score(X_test, y_test)   # 98% which is pretty good,

# Now for relHumid,
X_train, X_test, y_train, y_test = train_test_split(X, y_relHumid)

lin_alg.fit(X_train, y_train)
lin_alg.score(X_test, y_test) # 93%accurate

# now for absHumid
X_train, X_test, y_train, y_test = train_test_split(X, y_absHumid)

lin_alg.fit(X_train, y_train)
lin_alg.score(X_test, y_test) # 99.8% accurate waah! 

# these results were before preproscessinng the data.


