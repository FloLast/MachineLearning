import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib import rc

# Load the "gatlin" image data
X = np.loadtxt('gatlin.csv', delimiter=',')

#================= ADD YOUR CODE HERE ====================================
# Perform SVD decomposition
## TODO: Perform SVD on the X matrix
# Instructions: Perform SVD decomposition of matrix X. Save the 
#               three factors in variables U, S and V
#

U,S,V = np.linalg.svd(X)

#Xt = np.transpose(X)
#
#U = np.eigenvectors(X*Xt)
#V = np.eigenvectors(Xt*X)
#S = np.sqrt(np.eigenvalues(X*Xt))

#=========================================================================

# Plot the original image
plt.figure(1)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Matrix reconstruction using the top k = [10, 20, 50, 100, 200] singular values
## TODO: Create four matrices X10, X20, X50, X100, X200 for each low rank approximation
## using the top k = [10, 20, 50, 100, 200] singlular values 
#

print("U : " + str(U.shape))
print("S : " + str(S.shape))
print("V : " + str(V.shape))

X10 = np.dot(np.dot(U[:,:10],np.diag(S)[:10,:10]),V[:10,:])
X20 = np.dot(np.dot(U[:,:20],np.diag(S)[:20,:20]),V[:20,:])
X50 = np.dot(np.dot(U[:,:50],np.diag(S)[:50,:50]),V[:50,:])
X100 = np.dot(np.dot(U[:,:100],np.diag(S)[:100,:100]),V[:100,:])
X200 = np.dot(np.dot(U[:,:200],np.diag(S)[:200,:200]),V[:200,:])


#=========================================================================



#================= ADD YOUR CODE HERE ====================================
# Error of approximation
## TODO: Compute and print the error of each low rank approximation of the matrix
# The Frobenius error can be computed as |X - X_k| / |X|
#

print ("Error for k=10 ", (np.linalg.norm(X-X10) / np.linalg.norm(X)))
print ("Error for k=20 ", (np.linalg.norm(X-X20) / np.linalg.norm(X)))
print ("Error for k=50 ", (np.linalg.norm(X-X50) / np.linalg.norm(X)))
print ("Error for k=100 ", (np.linalg.norm(X-X100) / np.linalg.norm(X)))
print ("Error for k=200 ", (np.linalg.norm(X-X200) / np.linalg.norm(X)))

#=========================================================================



# Plot the optimal rank-k approximation for various values of k)
# Create a figure with 6 subfigures
plt.figure(2)

# Rank 10 approximation
plt.subplot(321)
plt.imshow(X10,cmap = cm.Greys_r)
plt.title('Best rank' + str(10) + ' approximation')
plt.axis('off')

# Rank 20 approximation
plt.subplot(322)
plt.imshow(X20,cmap = cm.Greys_r)
plt.title('Best rank' + str(20) + ' approximation')
plt.axis('off')

# Rank 50 approximation
plt.subplot(323)
plt.imshow(X50,cmap = cm.Greys_r)
plt.title('Best rank' + str(50) + ' approximation')
plt.axis('off')

# Rank 100 approximation
plt.subplot(324)
plt.imshow(X100,cmap = cm.Greys_r)
plt.title('Best rank' + str(100) + ' approximation')
plt.axis('off')

# Rank 200 approximation
plt.subplot(325)
plt.imshow(X200,cmap = cm.Greys_r)
plt.title('Best rank' + str(200) + ' approximation')
plt.axis('off')

# Original
plt.subplot(326)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (Rank 480)')
plt.axis('off')

plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Plot the singular values of the original matrix
## TODO: Plot the singular values of X versus their rank k

plt.figure(3)
plt.plot(S,linewidth=5)
plt.xlabel("rank k")


plt.draw()



#=========================================================================

plt.show() 

