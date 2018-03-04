#%%
import numpy as np
import pandas as pd
import pylab as plt
from time import sleep
from IPython import display


### Fetch the data and load it in pandas
data = pd.read_csv('train.csv')
print "Size of the data: ", data.shape

#%%
# See data (five rows) using pandas tools
print data.head()


### Prepare input to scikit and train and test cut

binary_data = data[np.logical_or(data['Cover_Type'] == 1,data['Cover_Type'] == 2)] # two-class classification set
X = binary_data.drop('Cover_Type', axis=1).values
y = binary_data['Cover_Type'].values
print np.unique(y)
y = 2 * y - 3 # converting labels from [1,2] to [-1,1]

#%%
# Import cross validation tools from scikit
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=None)


#%%
### Train a single decision tree

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=8)

# Train the classifier and print training time
clf.fit(X_train, y_train)

#%%
# Do classification on the test dataset and print classification results
from sklearn.metrics import classification_report
target_names = data['Cover_Type'].unique().astype(str).sort()
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))

#%%
# Compute accuracy of the classifier (correctly classified instances)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)




#===================================================================
#%%
### Train AdaBoost

# Your first exercise is to program AdaBoost.
# You can call *DecisionTreeClassifier* as above, 
# but you have to figure out how to pass the weight vector (for weighted classification) 
# to the *fit* function using the help pages of scikit-learn. At the end of 
# the loop, compute the training and test errors so the last section of the code can 
# plot the lerning curves. 
# 
# Once the code is finished, play around with the hyperparameters (D and T), 
# and try to understand what is happening.

D = 2 # tree depth
T = 400 # number of trees
w = np.ones(X_train.shape[0]) / X_train.shape[0]
training_scores = []
test_scores = []

ts = plt.arange(len(training_scores))
training_errors = []
test_errors = []
alpha = []
Yt = 0
Yt_test = 0

#===============================
for t in range(400):
    
    # Your code should go here
    ##3 Calling the base learner
    dt = DecisionTreeClassifier(max_depth = D)
    dt.fit(X_train,y_train,sample_weight=w)
    yt = dt.predict(X_train)
    yt_test = dt.predict(X_test)
    
    ##4 Computing the weighted error rate
    I = []
    for j in range(y_train.shape[0]):
        if yt[j] != y_train[j]:
            I.append(1)
        else :
            I.append(0)
    J = sum(w * I)
    gammat = J / sum(w)
    
    ##5
    alphat = np.log((1-gammat)/gammat)
    alpha.append(alphat)
    
    ##6
    for i in range(X_train.shape[0]):
        w[i] = w[i] * np.exp(alphat*I[i])
    
    
    
    Yt += alphat*yt
    training_scores.append(Yt)
    training_errors.append(np.mean(Yt-y_train))
    
    
    Yt_test += alphat*yt_test
#    for k in range(Yt_test.shape[0]):
#        if Yt_test[k] >= 0 :
#            Yt_test[k] = 1
#        else : 
#            Yt_test[k] = -1
    test_scores.append(Yt_test)
#    test_errors.append(1-accuracy_score(y_test,Yt_test))
    test_errors.append(np.mean(Yt_test-y_test))
    

#===============================

#  Plot training and test error    
plt.plot(training_errors, label="training error")
plt.plot(test_errors, label="test error")
plt.legend()



#===================================================================
#%%
### Optional part
### Optimize AdaBoost

# Your final exercise is to optimize the tree depth in AdaBoost. 
# Copy-paste your AdaBoost code into a function, and call it with different tree depths 
# and, for simplicity, with T = 100 iterations (number of trees). Plot the final 
# test error vs the tree depth. Discuss the plot.

#===============================

# Your code should go here
    

#===============================


