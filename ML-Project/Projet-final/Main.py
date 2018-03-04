# -*- coding: utf-8 -*-

import pandas as pd 
import matplotlib.pyplot as plt
from numpy import *
from tool import *
from datetime import *
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor



##############################################################################
#importation of data
data = pd.read_csv('uber_nyc_enriched.csv', delimiter = ";")
test = pd.read_csv('X_test.csv', delimiter = ";")


##############################################################################
#data vizualisation 

#histogram of spd column

#histo('spd',15,importation)

#pickup's mean in different borough

#for lieu in ['Bronx','Brooklyn','EWR','Manhattan','Queens','Staten Island']:
#print(lieu,' pickup's mean : ',mean(y[X[lieu]==1]))

#####################################################################################
#### Data preprocessing #############################################################
#####################################################################################
#get X the data and y the target (nbr of pickups)

#Setting our target 
y = data.pickups
y = array(y)

X = prepro(data)
del X['pickups']
X_test = prepro(test)


##############################################################################
#Features engeeneering 

#Creating features to fit polynomial regression based on the hour of different days
#here we set that the maximum degree of the polynomial is 5 
#polynomial_features = PolynomialFeatures(degree=5)

#polynomiale feature work that way :  
#For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]
#we apply this to hour and day of the week in order to have a polynomial for each day of the week 
#X_poly = polynomial_features.fit_transform(X[['hour','Workingday','Saturday', 'Sunday']])
#X2_poly = pd.DataFrame(X_poly)

#deleting all the columns that are empty (E.G. multiplying column Saturday with Sunday will only give 0)
#and deleting the columns that are similar 
#polyclean(X2_poly)

#deleting redonduant columns
#X=X.join(X2_poly)
#sup=['hour','Workingday','Saturday', 'Sunday']
#for i in sup: 
#    del X[i]

##############################################################################
### DIFFERENT ALGORITHMS #####################################################
##############################################################################
#Linear Regression normalized 
LR= LinearRegression(normalize=True)
#LR.fit(X,y)
#y_test = LR.predict(X_test)
#my_solution = pd.DataFrame(y_test)
#my_solution.to_csv("my_solution_LR.csv")

#total explained variance of the model : but with a big bias as their are huge difference between boroughs
print('Linear Regression R^2 : ' , mean(cross_val_score(LR,X,y,cv=10, scoring = 'r2')))
print('Linear Regression RMSE : ' , mean(sqrt(-cross_val_score(LR,X,y,cv=10, scoring = 'neg_mean_squared_error'))))


#so we build a model for each borough
#for lieu in ['Bronx','Brooklyn','EWR','Manhattan','Queens','Staten Island']:
#    X_new=X[X[lieu]==1]
#    y_new=y[X[lieu]==1]
#    X_new=dim_reduc(X_new,y_new,50)
#    print('Linear Regression', lieu, mean(cross_val_score(LR,X_new,y_new,cv=10,scoring='r2')))

#Staten island and EWR have very few pickups (max for EWR is 2 in an hour : no need for prediction)
#max de Staten Island is 13 pick ups in an hour (comparedn with 7883 in Manhattan) and the mean is close to one 

##############################################################################
#Lasso Regression normalized
lasso = Lasso(normalize=True, alpha = 0.01)
#lasso.fit(X,y)
#y_test = lasso.predict(X_test)
#my_solution = pd.DataFrame(y_test)
#my_solution.to_csv("my_solution_lasso.csv")
print('Lasso Regression R^2 : ' , mean(cross_val_score(lasso,X,y,cv=10, scoring = 'r2')))
print('Lasso Regression RMSE : ' , mean(sqrt(-cross_val_score(lasso,X,y,cv=10, scoring = 'neg_mean_squared_error'))))

##############################################################################
#Ridge Regression normalized
ridge = Ridge(normalize=True, alpha = 0.01)
#ridge.fit(X,y)
#y_test = ridge.predict(X_test)
#my_solution = pd.DataFrame(y_test)
#my_solution.to_csv("my_solution_ridge.csv")
print('Ridge Regression R^2 : ' , mean(cross_val_score(ridge,X,y,cv=10, scoring = 'r2')))
print('Ridge Regression RMSE : ' , mean(sqrt(-cross_val_score(ridge,X,y,cv=10, scoring = 'neg_mean_squared_error'))))

##############################################################################
#K-nearest neighbors
knn = KNeighborsRegressor(n_neighbors = 1, algorithm = 'brute', p = 1)
#knn.fit(X,y)
#y_test = knn.predict(X_test)
#my_solution = pd.DataFrame(y_test)
#my_solution.to_csv("my_solution_knn.csv")
print('1-Nearest Neighbor R^2 : ' , mean(cross_val_score(knn,X,y,cv=10, scoring = 'r2')))
print('1-Nearest Neighbor RMSE : ' , mean(sqrt(-cross_val_score(knn,X,y,cv=10, scoring = 'neg_mean_squared_error'))))

##############################################################################
#Decision Tree Regressor
dtr = DecisionTreeRegressor(min_samples_leaf=6, max_depth=10)
#dtr.fit(X,y)
#y_test = dtr.predict(X_test)
#my_solution = pd.DataFrame(y_test)
#my_solution.to_csv("my_solution_dtr.csv")
print('Decision Tree Regressor R^2 : ' , mean(cross_val_score(dtr,X,y,cv=10, scoring = 'r2')))
print('Decision Tree Regressor RMSE : ' , mean(sqrt(-cross_val_score(dtr,X,y,cv=10, scoring = 'neg_mean_squared_error'))))

##############################################################################
#AdaBoost Regressor
abr = AdaBoostRegressor(dtr,n_estimators =10)
#abr.fit(X,y)
#y_test = abr.predict(X_test)
#my_solution = pd.DataFrame(y_test)
#my_solution.to_csv("my_solution_abr.csv")
print('AdaBoost Regressor R^2 : ' , mean(cross_val_score(abr,X,y,cv=10, scoring = 'r2')))
print('AdaBoost Regressor RMSE : ' , mean(sqrt(-cross_val_score(abr,X,y,cv=10, scoring = 'neg_mean_squared_error'))))




















