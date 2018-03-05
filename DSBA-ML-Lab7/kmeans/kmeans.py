from numpy import *
from euclideanDistance import euclideanDistance
from simpleInitialization import simpleInitialization



def kmeans(X, k):
    # Intialize centroids
    centroids = simpleInitialization(X, k)
    
    #print(centroids)
    
    # Initialize variables
    iterations = 0
    oldCentroids = None
    labels = zeros(X.shape[0])
    
    # ====================== ADD YOUR CODE HERE ======================
    # Instructions: Run the main k-means algorithm. Follow the steps 
    #               given in the description. Compute the distance 
    #               between each instance and each centroid. Assign 
    #               the instance to the cluster described by the closest
    #               centroid. Repeat the above steps until the centroids
    #               stop moving or reached a certain number of iterations
    #               (e.g., 100).
    
    labels = []
    for i in range(100):
        c = []
        for j in range(len(centroids)):
            c.append(linalg.norm(X[i]-centroids[j]))
            labels += where(c == min(c))

    for point in range(len(X)):
        for j in range(len(centroids)):
            print X[labels[i]==j]
    # ===============================================================
        
    return labels
