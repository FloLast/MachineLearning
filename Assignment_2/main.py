from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import VotingClassifier,BaggingClassifier

from toolbox import *


# Load data
train = pd.read_csv('train.csv', sep = ',') # Load in the csv file
test = pd.read_csv('test.csv', sep = ',') # Load in the csv file
PassengerId = array(test["PassengerId"]).astype(int)
train.set_index('PassengerId',inplace=True, drop=True)
test.set_index('PassengerId',inplace=True, drop=True)
#print(train.head(10))
#print(train.count())

#Data for visualisation
plot_hist('Pclass',train)

##############################################################################
#Data preprocessing + Feature engineering
X,y = parse_model(train.copy())
X_kaggle = parse_model_test(test.copy())
print '*** Features : ***' 
print X.count()
#print(X_kaggle.count())

##############################################################################
#Cross validation train/test split
X, X_test, y, y_test = cross_validation.train_test_split(X,y, test_size=0.4, random_state=42)

##############################################################################
#Dimensionality reduction
X, X_test, X_kaggle = dim_reduc(X,X_test,X_kaggle,y,27)

##############################################################################
#Learning algorithm (LOGISTIC REGRESSION)
#lr = LogisticRegression()
#lr = lr.fit(X,y)
#print('Logistic Regression' , compute_score(lr,X,y))
#print lr.coef_

##############################################################################
#Learning algorithm (NEAREST NEIGHBORS)
#knc = KNeighborsClassifier(3)
#knc = knc.fit(X,y)
#print('K Nearest Neighbors', compute_score(knc,X,y))

##############################################################################
#Learning algorithm (SVM)
#svc1 = SVC(gamma=2, C=1)
#svc1 = svc1.fit(X,y)
#print('SVM without kernel' , compute_score(svc1,X,y))

##############################################################################
#Learning algorithm (SVM kernel)
#svc2 = SVC(kernel="linear", C=0.025)
#svc2 = svc2.fit(X,y)
#print('SVM with kernel' , compute_score(svc2,X,y))

##############################################################################
#Learning algorithm (NEURAL NETWORK)
#mlp = MLPClassifier(alpha=1)
#mlp = mlp.fit(X,y)
#print('Neural Network', compute_score(mlp,X,y))

##############################################################################
#Learning algorithm (DECISION TREES)
#dtc = DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
#dtc = dtc.fit(X, y)
#print('Decision Tree Classifier', compute_score(dtc,X,y))
#print(dtc.feature_importances_)

##############################################################################
#Learning algorithm (RANDOM FOREST) 
##Plotting the score given some different values of parameters
#random = []
#parameters = [50,80,100,120,150,180,200]
#for k in parameters:
#    rf = RandomForestClassifier(max_depth = 5, min_samples_split = 5, n_estimators = 180, random_state = 1)
#    rf = rf.fit(X,y)
#    random.append(compute_score(rf,X,y))
#plt.plot(parameters,random)
#prediction = rf.predict(X_kaggle)
#print('Random Forest Classifier' , compute_score(rf,X,y))

##Using gridsearch to find the best value of each parameter
#parameters = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}
#parameters = {'min_samples_split':[2,5,6,8]}
#parameters = {'n_estimators':[50,80,100,120,150,180,200]}
#parameters = {'warm_start':[True, False]}
#rf = RandomForestClassifier(warm_start=True, max_depth=5,n_estimators=180,min_samples_split=5,random_state=1)
#clf = GridSearchCV(rf, parameters)
#clf.fit(X, y)
#sorted(clf.cv_results_.keys())
#print('Random Forest Classifier' , compute_score(clf,X,y))
#print clf.best_estimator_
#prediction = clf.predict(X_kaggle)

rf = RandomForestClassifier(warm_start=True, max_depth=5,n_estimators=150,min_samples_split=6,random_state=1)
rf.fit(X, y)
print('Random Forest Classifier' , compute_score(rf,X,y))
prediction = rf.predict(X_kaggle)



##############################################################################
#Learning algorithm (BOOSTING)
#bst =  AdaBoostClassifier(n_estimators = 15)
#bst = bst.fit(X,y)
#print('AdaBoost', compute_score(bst,X,y))

##############################################################################
#bagging : creating random subset from the initial subset and applying an estimators
bcg = BaggingClassifier(base_estimator = dtc, n_estimators = 150, n_jobs=-1) 
bcg= bcg.fit(X,y)
prediction=bcg.predict(X_kaggle)

#Voting classifier : each model give a prediction, and get a vote (can be weigthed) : majority of vote give the predicion ('hard')
eclf = VotingClassifier(estimators=[('lr', lr), ('rf', rf),('dtc', dtc), ('bst', bst)], voting='hard')
eclf=eclf.fit(X,y)
prediction = eclf.predict(X_kaggle)

##############################################################################
#Evaluation

my_solution = pd.DataFrame(prediction, PassengerId, columns = ["Survived"])
#print(my_solution)
print(my_solution.shape)
my_solution.to_csv("my_solution_eight.csv", index_label = ["PassengerId"])






