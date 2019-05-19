# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:00:53 2019

@author: travis.hopkins
"""

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import *
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# load data set
df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

df_train.describe()


# explore dependent var --------------------
df_train['SalePrice'].describe()
# histogram
sns.distplot(df_train['SalePrice'])
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

# explore features that look promising based on description--------------------
# scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

# scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

# box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)

# explore all vars ----------------------------------------
# correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

# saleprice correlation matrix
# get only the 10 highest R values
k = 10  # number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
                 'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# removed vars that have multicollinearity

# scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea',
        'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)
plt.show()

# missing data -------------------------------
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()
           ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
# just checking that there's no missing data missing...
df_train.isnull().sum().max()

# standardizing data ----------------------------------
saleprice_scaled = StandardScaler().fit_transform(
    df_train['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)




# bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

# top two outliers follow trend => leave them
# ones on right do not => delete
# deleting points
df_train.sort_values(by='GrLivArea', ascending=False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

# bivariate analysis saleprice/GarageCars
var = 'GarageCars'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

# Sale Price has positive skewness and peaks high => apply log transformation
# applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])

# histogram and normal probability plot------------------------
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

# data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

# histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# create column for new variable (one is enough because it's a binary categorical feature)
# if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(
    len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

# transform data (just rows with HasBsmt == 1 can't log a zero)
df_train.loc[df_train['HasBsmt'] == 1,
             'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

# histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(
    df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)

# histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

# histogram and normal probability plot
sns.distplot(df_train['GarageCars'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GarageCars'], plot=plt)

# test homoscedasticity for two metric variables --------------------------------
# scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
# scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'],
            df_train[df_train['TotalBsmtSF'] > 0]['SalePrice'])


# convert categorical variable into dummy
df_train = pd.get_dummies(df_train)

#prepare test data set 
X_test = df_test[['GrLivArea', 'TotalBsmtSF',
                  'OverallQual', 'GarageCars']].values


# missing data in test -------------------------------
total = df_test[['GrLivArea', 'TotalBsmtSF', 'OverallQual',
                 'GarageCars']].isnull().sum().sort_values(ascending=False)
percent = (df_test[['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageCars']].isnull().sum(
)/df_test[['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageCars']].isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# dealing with missing data
df_test[df_test['TotalBsmtSF'].isnull()]
df_test[df_test['GarageCars'].isnull()]
#df_test = df_test.drop(df_test.index[660])
# #setting NaNs to 0; can't drop row due to Kaggle submition rules
where_are_NaNs = isnan(X_test)
X_test[where_are_NaNs] = 0

# select cols then convert to array MUCH EASIER for x (multiple cols)
Y = df_train['SalePrice'].values
X = df_train[['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageCars']].values


#train model #1 ------------------------------
model = LinearRegression()

model.fit(X, Y)

# add field names to the coeffs
names_2 = ['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageCars']
coeffs_zip = zip(names_2, model.coef_)
coeffs = set(coeffs_zip)

# print out fields with labels
print('Intercept: ', model.intercept_)

print('\n')

for coef in coeffs:
    print(coef, '\n')

# R2 of model
model.score(X, Y)


# predict prices with model on test data set
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
predicted_prices = model.predict(X_test)
print(predicted_prices)

# prepare Kaggle submission
my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('housingprice_submission_linearRegression.csv', index=False)

#model #2 ---------------------------------------------
# score: 0.99967379899475 
# with all X cols; seems to be overfitting
# need to fix dummy encoding a diff amount of cols for test and train sets

# #get dummies for categorical vars
# df_test = pd.get_dummies(df_test)

# # Get missing columns in the training test
# missing_cols = set( df_test.columns ) - set( df_train.columns )
# # Add a missing column in test set with default value equal to 0
# for c in missing_cols:
#     df_train[c] = 0
# # Ensure the order of column in the test set is in the same order than in train set
# df_train, df_test = df_train.align(df_test, axis=1)

# Y = df_train['SalePrice'].values
# #X = df_train[['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageCars']].values

# #get all cols expect SalePrice
# #X = df_train.drop('SalePrice', axis =1).values
# X = df_train.values

# X_test = df_test.values

# where_are_NaNs = isnan(X_test)
# X_test[where_are_NaNs] = 0

# model = RandomForestRegressor()

# model.fit(X, Y)

# model.score(X,Y)

# # predict prices with model on test data set
# predicted_prices = model.predict(X_test)
# print(predicted_prices)

# # prepare Kaggle submission
# my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': predicted_prices})
# # you could use any filename. We choose submission here
# my_submission.to_csv('housingprice_submission_decisionTree.csv', index=False)
