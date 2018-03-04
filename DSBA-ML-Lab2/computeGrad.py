from numpy import *
from sigmoid import sigmoid

def computeGrad(theta, X, y):
	# Computes the gradient of the cost with respect to
	# the parameters.
	
    m = X.shape[0] # number of training examples
    grad = zeros(size(theta)) # initialize gradient
    
    h=sigmoid(dot(X,theta))
    grad = dot(h - y,X)/m
    
#    for i in range(0,X.shape[1]) :
#        h=sigmoid(dot(X[i,:],theta))
#        grad = (1/m) * sum(dot((h - y),X))
#        i = i + 1
        
    return grad