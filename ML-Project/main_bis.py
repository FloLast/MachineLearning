#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:07:10 2017

@author: Flore
"""


from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


#%%

#####################################################################################
### FUNCTIONS #######################################################################
#####################################################################################
#Cross validation score
def compute_score(classifier,X,y):
    xval = cross_val_score(classifier,X,y,cv=10)
    return mean(xval)
#Dimension reduction
def dim_reduc(X,X_test,y,i):
    selector = SelectPercentile(f_classif, percentile=i)
    X = selector.fit_transform(X, y)
    X_test = selector.transform(X_test)    
    return X,X_test

#####################################################################################
### BEGINNING OF THE CODE ###########################################################
#####################################################################################
#Load the training data set
data = pd.read_csv('/Users/Flore/python/ML-Project/uber_nyc_enriched_2.csv', delimiter=';') 
y = data['pickups']
X = data
del X['pickups']

#%%

#plt.plot([array(X.pcp01), array(y)])

#%%

#####################################################################################
### Data preprocessing ##############################################################
#####################################################################################
#Replacing Nan by 0 for the added events
X['Event'] = X['Event'].fillna(0)

# Creating binary columns for boroughs and holiday days
borough_split = pd.get_dummies(X['borough'],prefix='borough')
X = X.join(borough_split, rsuffix = '1')
hday_split = pd.get_dummies(X['hday'],prefix='hday')
X = X.join(hday_split, rsuffix = '1')

# Creating three columns for month, day and hour
months = zeros(len(data))
days = zeros(len(data))
hours = zeros(len(data))
weekdays = zeros(len(data))
for i in range(len(data)):
    months[i] = dt.strptime(data['pickup_dt'][i], '%d/%m/%y %H:%M').strftime('%m')
    days[i] = dt.strptime(data['pickup_dt'][i], '%d/%m/%y %H:%M').strftime('%d')
    hours[i] = dt.strptime(data['pickup_dt'][i], '%d/%m/%y %H:%M').strftime('%H')
    weekdays[i] = dt.strptime(data['pickup_dt'][i], '%d/%m/%y %H:%M').strftime('%w')
months = pd.DataFrame(months, columns=['months']) 
days = pd.DataFrame(days, columns=['days']) 
hours = pd.DataFrame(hours, columns=['hours']) 
weekdays = pd.DataFrame(weekdays, columns=['weekdays']) 
X = X.join(months)
X = X.join(days)
X = X.join(hours)
X = X.join(weekdays)
X['is_evening'] = X.hours > 18
X['is_morning'] = X.hours < 3

#Deleting the redundant columns
to_del = ['borough', 'hday', 'pickup_dt']
for col in to_del : del X[col]

print X.head(5)

#%%

#####################################################################################
### Cross validation ################################################################
#####################################################################################
X, X_test, y, y_test = cross_validation.train_test_split(X,y, test_size=0.2, random_state=1)

#%%

#####################################################################################
### Dimensionality reduction ########################################################
#####################################################################################
print X
X, X_test, X_kaggle = dim_reduc(X,X_test,y,10)

#%%

#####################################################################################
### Training ########################################################################
#####################################################################################

#Ridge Regression
ridgereg = Ridge(normalize=True)
ridgereg.fit(X,y)
y_ridge = ridgereg.predict(X_test)
print('Ridge Regression' , compute_score(ridgereg,X,y))

#Lasso Regression
lasso = Lasso(normalize=True)
lasso.fit(X,y)
y_lasso = lasso.predict(X_test)
print('Lasso Regression' , compute_score(lasso,X,y))















