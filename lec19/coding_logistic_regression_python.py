# train a logistic regression classifier

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib.pyplpt as plt

iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris["data"])
# print(iris["target"])
# print(iris["DESCR"])

X = iris["data"][:, 3:]
Y = (iris["target"] == 2).astype(np.int)

#training classifier
clf = LogisticRegression()
clf.fit(X, Y)
example = clf.predict([[2.6]])
# print(example)

#using matplotlib

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
Y_prob = clf.predict_proba(X_new)

plt.plot(X_new, Y_prob[:, 1], "g-", Label="Verginica")
plt.show()
