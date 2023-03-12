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
from sklearn.model_selection import GridSearchCV

# Now, we will start to train logistic regression models on the Iris and Wine datasets.

# TODO: Load the Iris dataset using sklearn.
X, y = (load_breast_cancer().data, load_breast_cancer().target)
# Split train/test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 3)

# TODO: Initialize a logistic regression model for the Iris dataset.
# Here, you only need to tune the inverse regularization parameter `C`. 
# Please set `random_state` to 3.
max_samples_range = [260,265,266,267,268,269,270,271,272,273,274,275,280]
max_depth_range = [1,2,3,4,5,6, 7, 8, 9, 10]
parameters ={'max_samples':max_samples_range, 'max_depth':max_depth_range}
train_scores = []
test_scores = []
lr = RandomForestClassifier(random_state=3)
cv = GridSearchCV(lr, param_grid=parameters)
cv.fit(X_train, y_train)
print(cv.best_params_)
#lr.fit(X_train, y_train)