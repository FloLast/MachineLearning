# -*- coding: utf-8 -*-


import pandas as pd 
import matplotlib.pyplot as plt
from numpy import *
from datetime import *
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif


#Data visualization columns as data and number of bins as nbr

def histo(data,nbr,importation):
    fig, ax = plt.subplots()
    n, bins, patches =ax.hist(importation[data],nbr)
    ax.hist(n,bins,histtype='bar')
    ax.set_title('repartition')
    fig.tight_layout()
    plt.show()
    
#Data preprocessing : conv pickup_dt to month, week, day and location to dummies 

def prepro(importation):
    
    #Adding the hour to the data set
    dat=[]
    hour=[]
    for i in range(len(importation.pickup_dt)):
        dat.append(importation.pickup_dt[i].split(' ')[0])
        hour.append(importation.pickup_dt[i].split(' ')[1])
    hour=array(hour)
    for i in range(hour.shape[0]):
        hour[i]=int(hour[i].split(':')[0])
    
    importation['hour']=hour
    
    #Adding the date : the month and if it's a working day(monday-friday),a saturday or a sunday
    
    dat =array(dat)
    month=[]
    day=[]
    WD=[]
    m=0
    d=0
    for i in range(dat.shape[0]):
        m=int(dat[i].split('/')[1])
        month.append(m)
        d=int(dat[i].split('/')[0])
        day.append(d)
        WD.append(date(2015,m,d).weekday())

    
    WD =array(WD)
    Working=[]
    Sat=[]
    Sun=[]
    for k in range (WD.shape[0]):
        if WD[k]<5:
            Working.append(1)
            Sat.append(0)
            Sun.append(0)
        elif WD[k]==5 :
            Working.append(0)
            Sat.append(1)
            Sun.append(0)
        else :
            Working.append(0)
            Sat.append(0)
            Sun.append(1)
    Sat= array(Sat)
    Sun = array(Sun)
    Working=array(Working)
    
    importation['WeekDay']=WD
    importation['Workingday']=Working
    importation['Saturday']=Sat
    importation['Sunday']=Sun
    importation['month']=month
    
    #Obtaining dummues from the location of the pickups
   
    SITE = pd.get_dummies(importation.borough)
    
    importation=importation.join(SITE)
    
    #Converting the Holiday from Y and N to 1 and 0
    holiday_map={'N':0, 'Y':1}
    importation['hday']=importation['hday'].map(holiday_map)
    
    #filling the Event columns with data if its NaN : different from itself
    importation.loc[importation.Event != importation.Event,'Event']=0
    
    #deleting unnecessary columns
   
    del importation['pickup_dt']
    del importation['borough']
    
    return importation

###############################################################################
#polynomial cleaning 
def polyclean(X):
    for k in X.columns:
        if sum(X[k])==0:
            del X[k]
        else :
            redundant = False
            l=k+1
            while redundant==False :
                if l>max(X.columns):
                    redundant = True
                elif sum(X[k])==sum(X[l]):
                    redundant = True
                    del X[k]
                l=l+1

###############################################################################
#Crossvalidation 
def compute_score(classifier,X,y):
    xval = cross_val_score(classifier,X,y,cv=10)
    return mean(xval)

###############################################################################
#Dim reduction 

def dim_reduc(X,y,i):
    selector = SelectPercentile(f_classif, percentile=i)
    X = selector.fit_transform(X, y)  
    return X   

    
