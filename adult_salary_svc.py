# ----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# As dataset doesnot include the column headings, so using names for this
# dataset taken from : https://archive.ics.uci.edu/ml/datasets/Adult
# dataset also include missing which are denoted by ' ?' not '?'
dataset = pd.read_csv('Datasets/Adult_Sal.csv', names = ['age',
                                                         'workclass',
                                                         'fnlwgt',
                                                         'education',
                                                         'education-num',
                                                         'marital-status',
                                                         'occupation',
                                                         'relationship',
                                                         'race',
                                                         'sex',
                                                         'capital-gain',
                                                         'capital-loss',
                                                         'hours-per-week',
                                                         'native-country',
                                                         'salary'], na_values
                                                          = ' ?')

# after reading the vaues and fixing the ' ?' values with NaN
# first dividing X and y

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# Integer values index = 0,2,4,10,11,12
# String Values index = 1,3,5,6,7,8,9,13

from sklearn.preprocessing import Imputer

imp = Imputer()
X[:, [0, 2, 4, 10, 11, 12]] = imp.fit_transform(X[:, [0, 2, 4, 10, 11, 12]])

# Imputer is not best for using the string values , so we will use pandas
# first need to convert the object datatype to Dataframe
# Creating a test DataFrame object

test = pd.DataFrame(X[:, [1, 3, 5, 6, 7, 8, 9, 13]])
# Now, take one by one and value counts of each column and then replace NaN
# values by that mode
# For Workclass
test[0].value_counts()    # Most frequent = ' Private'
# For Education
test[1].value_counts()    # Most frequent = ' HS-grad'
# For marital Status
test[2].value_counts()    # Most frequent = ' Married-civ-spouse'
# For Occupation
test[3].value_counts()    # Most frequent = ' Prof-specialty'
# For Relationship
test[4].value_counts()    # Most frequent = ' Husband'
# For race
test[5].value_counts()    # Most frequent = ' White'
# For sex
test[6].value_counts()    # Most frequent = ' Male'
# For Native-Country
test[7].value_counts()    # Most frequent = ' United-States'


# Now we have to change each column with most frequent.
test[0] = test[0].fillna(' Private')
test[1] = test[1].fillna(' HS-grad')
test[2] = test[2].fillna(' Married-civ-spouse')
test[3] = test[3].fillna(' Prof-specialty')
test[4] = test[4].fillna(' Husband')
test[5] = test[5].fillna(' White')
test[6] = test[6].fillna(' Male')
test[7] = test[7].fillna(' United-States')

# Now reassign X from test
X[:, [1, 3, 5, 6, 7, 8, 9, 13]] = test
# Now delete test
# del(test)

# Now all the Missing values are handled now handle categorical values

from  sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()
# Label encoder works on each column one by one,
X[:, 1] = lab.fit_transform(X[:, 1])
X[:, 3] = lab.fit_transform(X[:, 3])
X[:, 5] = lab.fit_transform(X[:, 5])
X[:, 6] = lab.fit_transform(X[:, 6])
X[:, 7] = lab.fit_transform(X[:, 7])
X[:, 8] = lab.fit_transform(X[:, 8])
X[:, 9] = lab.fit_transform(X[:, 9])
X[:, 13] = lab.fit_transform(X[:, 13])

# After label encoding we have to do one hot encoding

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features= [1, 3, 5, 6, 7, 8, 9, 13])
X = one.fit_transform(X)
X = X.toarray()

# Now we have to use Standard Scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# After X we need to do y also, only use labelencoder on y
y = lab.fit_transform(y)
# ----------------------------------------------------------------------------
# ------------------------ABOVE FILE HAS BEEN COPIED FROM---------------------
# -------------------------adult_salary_preprocessing.py---------------------
# ---------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.svm import LinearSVC
lin_svc = LinearSVC()

lin_svc.fit(X_train, y_train)

lin_svc.score(X_test, y_test)
lin_svc.score(X_train, y_train)

from sklearn.svm import SVC
# For kernel = poly - test set - 82.2
svc= SVC(kernel = 'sigmoid')

svc.fit(X_train, y_train)

svc.score(X_test, y_test)
svc.score(X_train, y_train)

