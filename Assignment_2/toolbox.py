#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from numpy import *
from sklearn import cross_validation
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.cross_validation import cross_val_score

from sklearn import metrics
from timeit import default_timer as timer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression

#Data preprocessing + Feature engineering for the training set
def parse_model(X):    
    target = X.Survived
    
    X['title']= X.Name.map(lambda x : x.split(',')[1].split('.')[0])
    X['Cabin']= X.Cabin.map(lambda x : x[0] if not pd.isnull(x) else -1)
    X['Age'] = X.Age.fillna(X.Age.median())
    X['is_child'] = X.Age < 8
    X['family_size'] = X['SibSp'] + X['Parch'] + 1
    replace_titles(X)   
    replace_deck(X) 
    
    pclass_split = pd.get_dummies(X['Pclass'],prefix='Pclass')
    sex_split = pd.get_dummies(X['Sex'],prefix='Sex')
    title_split = pd.get_dummies(X['title'],prefix='title')
    cabin_split = pd.get_dummies(X['Cabin'],prefix='Cabin')
    embarked_split = pd.get_dummies(X['Embarked'],prefix='Embarked')
    cabin_split = pd.get_dummies(X['Cabin'],prefix='Cabin')
    X = X.join(pclass_split)
    X = X.join(sex_split)
    X = X.join(title_split)
    X = X.join(embarked_split)
    X = X.join(cabin_split)
    
    
    to_del = ['Name', 'title', 'Embarked','Survived','Cabin','Cabin_-1','Ticket','Sex','Pclass']
    for col in to_del : del X[col]
    return X,target

#Data preprocessing + Feature engineering for the test set
def parse_model_test(X):   
    
    X['title']= X.Name.map(lambda x : x.split(',')[1].split('.')[0])
    X['Cabin']= X.Cabin.map(lambda x : x[0] if not pd.isnull(x) else -1)
    X['Age'] = X.Age.fillna(X.Age.median())
    X['Fare'] = X.Fare.fillna(X.Fare.median())
    X['is_child'] = X.Age < 8
    X['family_size'] = X['SibSp'] + X['Parch'] + 1
    replace_titles(X) 
    replace_deck(X) 
    
    
    
    pclass_split = pd.get_dummies(X['Pclass'],prefix='Pclass')
    sex_split = pd.get_dummies(X['Sex'],prefix='Sex')
    title_split = pd.get_dummies(X['title'],prefix='title')
    cabin_split = pd.get_dummies(X['Cabin'],prefix='Cabin')
    embarked_split = pd.get_dummies(X['Embarked'],prefix='Embarked')
    cabin_split = pd.get_dummies(X['Cabin'],prefix='Cabin')
    X = X.join(pclass_split)
    X = X.join(sex_split)
    X = X.join(title_split)
    X = X.join(embarked_split)
    X = X.join(cabin_split)
    
    
    to_del = ['Name', 'title', 'Embarked','Ticket','Cabin','Sex','Pclass']
    for col in to_del : del X[col]
    return X

#Implementation of k-fold cross validation
def compute_score(classifier,X,y):
    xval = cross_val_score(classifier,X,y,cv=10)
    return mean(xval)

#For data visualisation
def plot_hist(feature, train, bins = 20):
    survived = train[train.Survived == 1]
    dead = train[train.Survived == 0]
    x1 = array(dead[feature].dropna())
    x2 = array(survived[feature].dropna())
    plt.hist([x1,x2],label=['Deads','Survivors'], bins=bins)
    plt.legend(loc='upper left')
    plt.title('Relative Distribution of %s' %feature)
    plt.show()
    
#For dimension reduction
def dim_reduc(X,X_test,X_kaggle,y,nb):
    ch2 = SelectKBest(chi2,k=nb)
    X = ch2.fit_transform(X, y)
    X_test = ch2.transform(X_test)
    X_kaggle = ch2.transform(X_kaggle)
    return X,X_test,X_kaggle

#Special function to regroup the titles by type
def replace_titles(X):
    X['title'] = X['title'].str.replace('Mlle','Miss')
    X['title'] = X['title'].str.replace('Mme','Mrs') 
    X['title'] = X['title'].str.replace('Master','Mr')
    X['title'] = X['title'].str.replace('Capt','Milit')      
    X['title'] = X['title'].str.replace('Col','Milit')
    X['title'] = X['title'].str.replace('Lady','Nobles')
    X['title'] = X['title'].str.replace('Sir','Nobles')
    X['title'] = X['title'].str.replace('the Countess','Nobles')
    X['title'] = X['title'].str.replace('Dona','Nobles')
    X['title'] = X['title'].str.replace('Don','Nobles')
    X['title'] = X['title'].str.replace('Jonkheer','Nobles')
    X['title'] = X['title'].str.replace('Major','Job')
    X['title'] = X['title'].str.replace('Dr','Job')
    X['title'] = X['title'].str.replace('Rev','Job')
    return X

def replace_deck(X):
    X['Cabin'] = X['Cabin'].str.replace('T','-1')
    return X

