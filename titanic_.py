# Titanic dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# This dataset is taken from :
# https://www.kaggle.com/c/titanic/data
dataset = pd.read_csv("Datasets/titanic_train.csv")
gender = pd.read_csv("Datasets/titanic_gender_submission.csv")

dataset.merge(gender, left_on='PassengerId', right_on='PassengerId')

X = dataset.iloc[:, [2,4,5,6,7,9,11]]
y = dataset.iloc[:, 1]
  