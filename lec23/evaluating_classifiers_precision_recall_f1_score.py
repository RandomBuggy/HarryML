import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#below imports are for lec-23
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

mnist = fetch_openml("mnist_784")
x, y = mnist["data"], mnist["target"]
x.shape
y.shape
# %matplot inline

some_digit = x[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")

# train-test-split
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

# shuffling train data
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

# creating a 2 detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)


# binary classifier
clf = LogisticRegression(tol = 0.1, solver = "lbfgs")
clf.fit(x_train, y_train_2)
clf.predict(some_digit)

cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")



# lec-23 precision, recall & F1 score, confusion matrix
y_train_predict = cross_val_predict(clf, x_train, y_train_2, cv=3)

# confusion matrix
confusion_matrix(y_train_2, y_train_predict)
confusion_matrix(y_train_2, y_train_2) # perfect confusion_matrix

# precision and recall

precision_score(y_train_2, y_train_predict)
recall_score(y_train_2, y_train_predict)

# F1-score

f1_score(y_train_2, y_train_predict)

# precision-recall curve
y_scores = cross_val_predict(clf, x_train, y_train_2, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_2, y_scores)


#plotting
plt.plot(thresholds, precisions[:-1], "b--", label="precision")
plt.plot(thresholds, recalls[:-1], "g-", label="recall")
plt.xlabel("Thresholds")
plt.legend(loc="upper left")
plt.ylim([0, 1])
plt.show()
