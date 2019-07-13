import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# This dataset has been downloaded from :
# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
# The data is not comma seperated, instead its semicolon seperated
dataset=pd.read_csv("Datasets/bank-additional-full.csv", sep=';')
# dataset.isnull().values.any() Check for NaN values -- False


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

pd.plotting.scatter_matrix(dataset)


# Integer Cols - 0, 10, 11, 12, 13, 15, 16, 17, 18, 19
# Categorical Cols - 1,2,3,4,5,6,7,8,9,14,

# No Imputer as cannot detect missing values
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

X[:, 1]=lab.fit_transform(X[:, 1])
X[:, 2]=lab.fit_transform(X[:, 2])
X[:, 3]=lab.fit_transform(X[:, 3])
X[:, 4]=lab.fit_transform(X[:, 4])
X[:, 5]=lab.fit_transform(X[:, 5])
X[:, 6]=lab.fit_transform(X[:, 6])
X[:, 7]=lab.fit_transform(X[:, 7])
X[:, 8]=lab.fit_transform(X[:, 8])
X[:, 9]=lab.fit_transform(X[:, 9])
X[:, 14]=lab.fit_transform(X[:, 14])
y = lab.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features= [1,2,3,4,5,6,7,8,9,14])

X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(X)

# Preprocessing complete - 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()

dtf.fit(X_train, y_train)
dtf.score(X_test, y_test) # 88.62%
dtf.score(X_train, y_train) # 1
dtf.score(X, y) # 97.15%


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

log_reg.score(X_test, y_test) #  91.05%
log_reg.score(X_train, y_train) # 91.24%
log_reg.score(X, y) # 91.19%


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn.score(X_test, y_test) # 89.44%
knn.score(X_train, y_train) # 92.05%
knn.score(X, y)

from sklearn.svm import SVC
svc = SVC()

svc.fit(X_train, y_train)

svc.score(X_test, y_test) # 91% 
svc.score(X_train, y_train) # 92.8% 
# So in this case svc is a good model
 
