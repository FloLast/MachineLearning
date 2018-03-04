# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 17:16:10 2017

@author: Hugo
"""

import pandas as pd 
import matplotlib.pyplot as plt
from numpy import *
from tool import *
from datetime import *
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import PolynomialFeatures


##############################################################################
#importation of data
importation = pd.read_csv('uber_nyc_enriched.csv', delimiter = ";")


##############################################################################
#data vizualisation 

#histogram of spd column

#histo('spd',15,importation)

#pickup's mean in different borough

#for lieu in ['Bronx','Brooklyn','EWR','Manhattan','Queens','Staten Island']:
#    print(lieu, "pickup's mean :",mean(y[X[lieu]==1]))

##############################################################################
#Data preprocessing

#get X the data and y the target (nbr of pickups)

X,y = prepro(importation)

##############################################################################
#Features engeeneering 

#Creating features to fit polynomial regression based on the hour of different days
#here we set that the maximum degree of the polynomial is 5 

polynomial_features = PolynomialFeatures(degree=5)

#polynomiale feature work that way :  
#For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]
#we apply this to hour and day of the week in order to have a polynomial for each day of the week 

X_poly=polynomial_features.fit_transform(X[['hour','Workingday','Saturday', 'Sunday']])
X2_poly=pd.DataFrame(X_poly)

#deleting all the columns that are empty (E.G. multiplying column Saturday with Sunday will only givce 0)

for k in X2_poly.columns:
    if sum(X2_poly[k])==0:
        del X2_poly[k]

#deleting redonduant columns

X=X.join(X2_poly)
sup=['hour','Workingday','Saturday', 'Sunday']
for i in sup: 
    del X[i]

##############################################################################
### Different Algorithms #####################################################
##############################################################################
#Model
#Linear Regression normalized 
LR= LinearRegression(normalize=True)

#total explained variance of the model : but with a big bias as their are huge difference between boroughs
print('Linear Regression', mean(cross_val_score(LR,X,y,cv=10)))


#so we build a model for each borough
for lieu in ['Bronx','Brooklyn','EWR','Manhattan','Queens','Staten Island']:
    X_new=X[X[lieu]==1]
    y_new=y[X[lieu]==1]
    X_new=dim_reduc(X_new,y_new,50)
    print('Linear Regression', lieu, mean(cross_val_score(LR,X_new,y_new,cv=10)))

#Staten island and EWR have very few pickups (max for EWR is 2 in an hour : no need for prediction)
#max de Staten Island is 13 pick ups in an hour (comparedn with 7883 in Manhattan) and the mean is close to one 


    
    
    
    
    
    
    
    
    
    
    