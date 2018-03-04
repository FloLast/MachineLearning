"""
Created on Mon Oct 23 11:28:54 2017
@author: Flore
"""

from numpy import *
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from timeit import default_timer as timer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectPercentile, f_classif


############ QUESTION 1 #################

#Load the training data set
data = loadtxt('data.csv', delimiter=',')
y = array(data[:,0:1])
X = data[: ,1:data.shape[1]]
print(X.shape)

#Load the test data set
test = loadtxt('test.csv', delimiter=',')
y_test = array(test[:,0:1])
X_test = test[: ,1:test.shape[1]]

#Training of the data set
start = timer()
logistic = LogisticRegression()
logistic.fit(X,y)
pred = logistic.predict(X)
end = timer()

#ROC curve
y_scores = logistic.fit(X,y).decision_function(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores )
plt.figure(1)
plt.title('ROC curve of the test set')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.plot(fpr, tpr, 'r')
plt.show()

#Area under the curve
auc = metrics.roc_auc_score(y_test, y_scores)
print("Area under the curve : " + str(auc))

#Required running time for training
print("Required time for training : " + str(end-start) + " seconds")

############ QUESTION 2 #################

#For chi2 we need non negative X :
cst = X.min()
X_a = X + abs(cst)
X_test_a = X_test + abs(cst)

#Applying the following steps for the range {20, 40, 60, 80, 100, 150}
nbFeatures = [20,40,60,80,85,90,95,100,150]
areas_under_curve = []
timers = []
for i in nbFeatures :

    ##Feature selection 
    ch2 = SelectKBest(chi2, k=i)
    X_k = ch2.fit_transform(X_a, y)
    X_test_k = ch2.transform(X_test_a)
    
    ##Training 
    start_k = timer()
    logistic = LogisticRegression()
    logistic.fit(X_k,y)
    end_k = timer()
    
    ##Computing area under the curve & time 
    ###Area under the curve
    y_scores_k = logistic.fit(X_k,y).decision_function(X_test_k)
    auc_k = metrics.roc_auc_score(y_test, y_scores_k)
    ###Required running time for training
    time_k = end_k-start_k
    ###Adding each new area/time to their array
    areas_under_curve.append(auc_k)
    timers.append(time_k)
    
##Plotting area under the curve VS nb of features + time VS nb of features
plt.figure(2)
plt.subplot(211)
plt.plot(nbFeatures,areas_under_curve)
plt.xlabel('Number of features')
plt.ylabel('Area under curve')
plt.subplot(212)
plt.plot(nbFeatures,timers)
plt.xlabel('Number of features')
plt.ylabel('Time for training')
plt.show()

############ QUESTION 3 #################


#Applying the following steps to different percentiles
percentiles = [10,20,30,40,50,55,60,65,70,80,90]
areas_under_curve_new = []
timers_new = []
for i in percentiles :

    ##Feature selection with the SelectPercentile algorithm
    selector = SelectPercentile(f_classif, percentile=i)
    X_new = selector.fit_transform(X, y)
    X_test_new = selector.transform(X_test)
    
    ##Training
    start_new = timer()
    logistic = LogisticRegression()
    logistic.fit(X_new,y)
    end_new = timer()
    
    ##Computing area under the curve & time 
    ###Area under the curve
    y_scores_new = logistic.fit(X_new,y).decision_function(X_test_new)
    auc_new = metrics.roc_auc_score(y_test, y_scores_new)
    ###Required running time for training
    time_new = end_new-start_new
    ###Adding each new area/time to their array
    areas_under_curve_new.append(auc_new)
    timers_new.append(time_new)

#Plotting area under the curve VS Percent of features kept 
#+ time VS Percent of features kept
plt.figure(3)
plt.subplot(311)
plt.plot(percentiles,areas_under_curve_new)
plt.xlabel('Percent of features kept')
plt.ylabel('Area under curve')
plt.subplot(312)
plt.plot(percentiles,timers_new)
plt.xlabel('Percent of features kept')
plt.ylabel('Time for training')
plt.show()
