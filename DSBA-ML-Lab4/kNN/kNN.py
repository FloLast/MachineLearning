from numpy import *
from euclideanDistance import euclideanDistance

def kNN(k, X, labels, y):
    # Assigns to the test instance the label of the majority of the labels of the k closest 
	# training examples using the kNN with euclidean distance.
    #
    # Input: k: number of nearest neighbors
    #        X: training data           
    #        labels: class labels of training data
    #        y: test data
    
    
    # ====================== ADD YOUR CODE HERE =============================
    # Instructions: Run the kNN algorithm to predict the class of
    #               y. Rows of X correspond to observations, columns
    #               to features. The 'labels' vector contains the 
    #               class to which each observation of the training 
    #               data X belongs. Calculate the distance betweet y and each 
    #               row of X, find  the k closest observations and give y 
    #               the class of the majority of them.
    #
    # Note: To compute the distance betweet two vectors A and B use
    #       use the euclideanDistance(A,B) function.
    #
    
    
    d = []
    #print(X[1,:].shape)
    #print(y.shape)
    for i in range(X.shape[0]):
        d.append(euclideanDistance(X[i,:],y))
        
    d = array(d)
    labels = array(labels)
    
    idx = d.argsort()[::-1] # Sort distances
    labels = labels[idx] # Sort labels according to distances    
    
    selected = labels[-k:]
    #print(selected)

    count = {}.fromkeys(set(selected),0)
    for value in selected:
        count[value] += 1
    maximum = max(count, key=count.get)
    
    label = int(maximum)
    print(label)

    # return the label of the test data
    return label

 
