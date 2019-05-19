"""
Created on May 17 2019

@author: travis.hopkins
"""

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# load data set
zoo = pd.read_csv('./zoo.csv')

# EDA
zoo.dtypes
zoo.describe()

# explore dependent var
zoo.groupby('class_type').count()

# explore all vars
# correlation matrix
corrmat = zoo.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

# check for missing data
total = zoo.isnull().sum().sort_values(ascending=False)
percent = (zoo.isnull().sum(
)/zoo.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# saleprice correlation matrix
# get only the 10 highest R values
k = 20  # number of variables for heatmap
cols = corrmat.nlargest(k, 'class_type')['class_type'].index
cm = np.corrcoef(zoo[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
                 'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()



# MODEL ONE
# ----------------------
# Accuracy: 97.059%
# Accuracy: 0.960 (0.066)

# split data into test and train
Y = zoo['class_type']
X = zoo[['catsize','hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic']]
#X = zoo[['catsize', 'eggs', 'hair', 'milk', 'predator',
         #'toothed', 'backbone', 'breathes', 'tail']]

# set test size and seed
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
                                                    random_state=seed)

train_test_split(zoo)
# train decision tree model with train data
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

#visualize tree
# dot_data = StringIO()
# export_graphviz(model, out_file=dot_data,  filled=True, rounded=True,   special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(graph.create_png())

# classification report / scoring
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)

result = model.score(X_test, Y_test)

print(("Accuracy: %.3f%%") % (result*100.0))

# 10 fold cross validation
n_splits = 10
seed = 7
kfold = KFold(n_splits, random_state=seed)
scoring = 'accuracy'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))

# MODEL TWO
# ----------------------
#with all fields
# Accuracy: 88.235%
# Accuracy: 0.890 (0.137)

# after selecting cols
# Accuracy: 94.118%
# Accuracy: 0.900 (0.077)

# split data into test and train
# REMOVING FIELDS WITH LOW CORRELATION 
# and multicollinearity (hair/milk: 0.88) for the KNN model
Y = zoo['class_type']
X = zoo[['catsize','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic', 'hair']]
#X = zoo[['catsize', 'eggs', 'milk', 'predator','toothed', 'backbone', 'breathes', 'tail']]

# set test size and seed
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
                                                    random_state=seed)
train_test_split(zoo)

# train KNN model with train data
model = KNeighborsClassifier()
model.fit(X_train, Y_train)

# classification report / scoring
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)

result = model.score(X_test, Y_test)

print(("Accuracy: %.3f%%") % (result*100.0))

# 10 fold cross validation
n_splits = 10
seed = 7
kfold = KFold(n_splits, random_state=seed)
scoring = 'accuracy'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
