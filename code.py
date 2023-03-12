#--- Load packages for datasets---
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

#--- Load packages for logistic regression and random forest---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#--- Load packages for train/test split---
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

# Now, we will start to train logistic regression models on the Iris and Wine datasets.

# TODO: Load the Iris dataset using sklearn.
X, y = (load_iris().data, load_iris().target)
# Split train/test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 3)

# TODO: Initialize a logistic regression model for the Iris dataset.
# Here, you only need to tune the inverse regularization parameter `C`. 
# Please set `random_state` to 3.
c_range = [0.0001, 0.0005, 0.001, 0.003, 0.006, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
train_errors = []
test_errors = []
for c in c_range:
    lr = LogisticRegression(random_state=3, C=c, max_iter=10000)

    # Start training.
    lr.fit(X_train, y_train)

    train_errors.append(1-lr.score(X_train, y_train))
    test_errors.append(1-lr.score(X_test, y_test))

plt.plot(c_range, train_errors, color='red')
plt.plot(c_range, test_errors, '--', color='blue')
plt.plot(c_range, np.subtract(test_errors, train_errors), color='green')

plt.ylabel('error')
plt.xlabel('C range')
plt.show()

# TODO: Load the Wine dataset.
X, y = (load_wine().data, load_wine().target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 1)
c_range = [0.0001, 0.001, 0.006, 0.01, 0.02, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.4, 0.5, 0.6, 0.7]
# TODO: Initialize a logistic regression model.
# Here, you only need to tune the inverse regularization parameter `C`. 
# Please set `random_state` to 3.
train_errors = []
test_errors = []
for c in c_range:
    lr = LogisticRegression(random_state=3, C=c, max_iter=10000)

    # Start training.
    lr.fit(X_train, y_train)

    train_errors.append(1-lr.score(X_train, y_train))
    test_errors.append(1-lr.score(X_test, y_test))

plt.plot(c_range, train_errors, color='red')
plt.plot(c_range, test_errors, '--', color='blue')
plt.plot(c_range, np.subtract(test_errors, train_errors), color='green')

plt.ylabel('error')
plt.xlabel('C range')
plt.show()

# Now, we will start to train random forest models on the Iris and Breast Cancer datasets.

# Load the Iris dataset for training a random forest model.
X, y = (load_iris().data, load_iris().target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 1)

# Initialize a random forest model using sklearn.
# Here, you need to take turns to tune max_depth/max_samples for showing cases of underfitting/overfitting.
# Note that when you tune max_depth, please leave max_samples unchanged!
# Similarly, when you tune max_samples, leave max_depth unchanged!
# Please set `random_state` to 3 and feel free to set the value of `n_estimators`.
train_errors = []
test_errors = []

max_depth_range = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50]

for d in max_depth_range:
    rf = RandomForestClassifier(random_state=3,max_depth=d, max_samples=12)
    # Start training.
    rf.fit(X_train, y_train)

    train_errors.append(1-rf.score(X_train, y_train))
    test_errors.append(1-rf.score(X_test, y_test))

plt.plot(max_depth_range, train_errors, color='red')
plt.plot(max_depth_range, test_errors, '--', color='blue')
plt.plot(max_depth_range, np.subtract(test_errors, train_errors), color='green')

plt.ylabel('error')
plt.xlabel('max depth range')
plt.show()

train_errors = []
test_errors = []
max_samples_range = [1, 3, 5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
for s in max_samples_range:
    rf = RandomForestClassifier(random_state=3,max_depth=30, max_samples=s)
    # Start training.
    rf.fit(X_train, y_train)

    train_errors.append(1-rf.score(X_train, y_train))
    test_errors.append(1-rf.score(X_test, y_test))

plt.plot(max_samples_range, train_errors, color='red')
plt.plot(max_samples_range, test_errors, '--', color='blue')
plt.plot(max_samples_range, np.subtract(test_errors, train_errors), color='green')

plt.ylabel('error')
plt.xlabel('max samples range')
plt.show()

# Load the Breast Cancer dataset.
X, y = (load_breast_cancer().data, load_breast_cancer().target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 3)

train_errors = []
test_errors = []

max_depth_range = [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 50]

# TODO: Initialize a random forest model for the Breast Cancer dataset.
for d in max_depth_range:
    rf = RandomForestClassifier(random_state=3,max_depth=d, max_samples=270)
    # Start training.
    rf.fit(X_train, y_train)

    train_errors.append(1-rf.score(X_train, y_train))
    test_errors.append(1-rf.score(X_test, y_test))

plt.plot(max_depth_range, train_errors, color='red')
plt.plot(max_depth_range, test_errors, '--', color='blue')
plt.plot(max_depth_range, np.subtract(test_errors, train_errors), color='green')

plt.ylabel('error')
plt.xlabel('max depth range')
plt.show()

train_errors = []
test_errors = []
max_samples_range = [1, 5, 10, 30, 50, 80, 100, 150, 170, 200, 230, 240, 250, 260, 270, 280, 290, 300]
for s in max_samples_range:
    rf = RandomForestClassifier(random_state=3,max_depth=30, max_samples=s)
    # Start training.
    rf.fit(X_train, y_train)

    train_errors.append(1-rf.score(X_train, y_train))
    test_errors.append(1-rf.score(X_test, y_test))

plt.plot(max_samples_range, train_errors, color='red')
plt.plot(max_samples_range, test_errors, '--', color='blue')
plt.plot(max_samples_range, np.subtract(test_errors, train_errors), color='green')

plt.ylabel('error')
plt.xlabel('max samples range')
plt.show()
