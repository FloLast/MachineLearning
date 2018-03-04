from sigmoid import sigmoid
import numpy as np

def computeCost(theta, X, y): 
	# Computes the cost using theta as the parameter 
	# for logistic regression. 
    
    m = X.shape[0] # number of training examples
    J = 0
    
    h=sigmoid(np.dot(X,theta))
    J = - sum(y*np.log(h)+(1-y)*np.log(1-h))/m
#    
#    for i in range(0,X.shape[1]) :
#        h=sigmoid(dot(X[i,:],theta))
#        J = (1/m) * sum(y*log(h)+(1-y)*log(1-h))
#        i = i + 1
#    
    return J


