import math
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load data (Wine dataset)
my_data = np.genfromtxt('wine_data.csv', delimiter=',')
np.random.shuffle(my_data) # shuffle datataset
trainingData = my_data[:100,1:] # training data
trainingLabels = my_data[:100,0] # class labels of training data

testData = my_data[101:,1:] # training data
testLabels = my_data[101:,0] # class labels of training data

def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """    

    
    classLabels = np.unique(Y) # different class labels on the dataset
    classNum = len(classLabels)
    datanum, dim = X.shape # dimensions of the dataset
    totalMean = np.mean(X,0) # total mean of the data

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the LDA technique, following the
    # steps given in the pseudocode on the assignment.
    # The function should return the projection matrix W,
    # the centroid vector for each class projected to the new
    # space defined by W and the projected data X_lda.    
      
        
    #1/ WITHIN CLASS SCATTER MATRIX
    sw = 0
    for j in range(classNum):
        Xj = X[np.where(Y==classLabels[j]),:] 
        mj = np.mean(Xj,1)
        for i in range(dim):
            a = Xj[0,:,:]-mj
            b = np.transpose(a)
            c = np.dot(b,a)
            sw = sw + c
 
            
    #2/ BETWEEN CLASS SCATTER MATRIX
    st = np.cov(np.transpose(X))
    sb = st - sw

    #3/ MATRIX W WITH A PCA
    C = np.dot(sw**(-1),sb)
    
    M = np.mean(C,0)
    D = C - M
    
    W0 = np.dot(np.transpose(D),D)
    eigval, eigvec = linalg.eig(W0)
    
    idx = eigval.argsort()[::-1]
    eigvec = eigvec[:,idx]

    E = eigval[:-1]
    W = eigvec[:,:-1]    
    X_lda = np.dot(X,W)
    
    projected_centroid = 0
    for j in range(classNum):
        mj = np.mean(Xj,1)
        projected_centroid += np.dot(mj,W)
    
    # =============================================================

    return W, projected_centroid, X_lda


# Training LDA classifier
W, projected_centroid, X_lda = my_LDA(trainingData, trainingLabels)

# Perform predictions for the test data
predictedLabels = predict(testData, projected_centroid, W)
predictedLabels = predictedLabels+1


# Compute accuracy
counter = 0
for i in range(predictedLabels.size):
    if predictedLabels[i] == testLabels[i]:
        counter += 1
print 'Accuracy of LDA: %f' % (counter / float(predictedLabels.size) * 100.0)

