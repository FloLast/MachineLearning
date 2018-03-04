import numpy as np
import scipy as sp
import scipy.linalg as linalg

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

    #3/ MATRIX W 
    C = np.dot(np.linalg.inv(sw),sb)
    M = np.mean(C,0)
    D = C - M
    W0 = np.dot(np.transpose(D),D)
    
    ## PCA
    eigval, eigvec = linalg.eig(W0)
    idx = eigval.argsort()[::-1]
    eigvec = eigvec[:,idx]    
    W = eigvec[:,:-1] 
    
    ## SVD
#    U, S, V = np.linalg.svd(W0)
#    W = U[:,:-1]
    
    #3/ PROJECT TO NEW SPACE AND COMPUTE X_lda
    X_lda = np.dot(X,W)
    
    #4/ PROJECT EACH VECTOR OF EACH CLASS AND COMPUTE projected_centroid
#    projected_centroid = 0
#    for j in range(classNum):
#        Xj = X[np.where(Y==classLabels[j]),:] 
#        mj = np.mean(Xj,1)
#        projected_centroid += np.dot(mj,W)
    partition = [np.where(Y==label)[0] for label in classLabels]
    classMean = [(np.mean(X[idx],0),len(idx)) for idx in partition]
    projected_centroid = [np.dot(mu, np.real(W)) for mu,class_size in classMean]
    
    # =============================================================

    return W, projected_centroid, X_lda