from numpy import *
from euclideanDistance import euclideanDistance

def findClosestNeighbours(data, N):
    
    closestNeighbours = zeros((data.shape[0], N))
    distances = zeros(data.shape[0])

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i != j:
                distances[j] = euclideanDistance(data[i,:], data[j,:])    
            else:
                distances[j] = 0
                
        closestNeighbours[i,:] = argsort(distances)[:N]
    
    return closestNeighbours
