import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples = 300, centers = 5, cluster_std=0.7)

plt.scatter(x[:, 0], x[:,1])
plt.show()

from sklearn.cluster import KMeans

# Change k from 1-14
wcv = []


for i in range(1, 15):
  km = KMeans(n_clusters=i)
  km.fit(x)
  wcv.append(km.inertia_)

# The elbow point is 5
plt.plot(range(1,15), wcv)
plt.show()

# Now using n_clusters = 5
km = KMeans(n_clusters=5)
y_pred = km.fit_predict(x)


