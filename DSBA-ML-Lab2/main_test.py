import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def computeCost(theta, X, y): 
    m = X.shape[0] # number of training examples
    J = 0
    h=sigmoid(np.dot(X,theta))
    J = - sum(y*np.log(h)+(1-y)*np.log(1-h))/m
    #print(" J : " + str(J))
    return J

def computeGrad(theta, X, y):
    m = X.shape[0] # number of training examples
    grad = np.zeros(np.size(theta)) # initialize gradient
    grad = np.dot((sigmoid(np.dot(X,theta)) - y),X)/m
    #print("grad : " + str(grad))
    return grad

# Load the dataset
# The first two columns contains the exam scores and the third column
# contains the label.
data = np.loadtxt('data1.txt', delimiter=',')
 
X = data[:, 0:2]
y = data[:, 2]

# Plot data 
pos = np.where(y == 1) # instances of class 1
neg = np.where(y == 0) # instances of class 0
plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not Admitted'])
plt.show()


#Add intercept term to X
X_new = np.ones((X.shape[0], 3))
X_new[:, 1:3] = X
X = X_new

# Initialize fitting parameters
initial_theta = np.zeros((3,1))

#c = computeCost(initial_theta,X,y)
#g = computeGrad(initial_theta,X,y)
#print("ComputeCost : " + str(c.shape))
#print("ComputeGrad : " + str(g.shape))

# Run minimize() to obtain the optimal theta
Result = op.minimize(fun = computeCost, x0 = initial_theta, args = (X, y), method = 'TNC',jac = computeGrad);
print(Result)
theta = Result.x;
print(theta)

# Plot the decision boundary
plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 2]) + 2])
plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
plt.plot(plot_x, plot_y)
plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='b')
plt.scatter(X[neg, 1], X[neg, 2], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Decision Boundary', 'Admitted', 'Not Admitted'])
plt.show()

# Compute accuracy on the training set
p = predict(array(theta), X)
counter = 0
for i in range(y.size):
    if p[i] == y[i]:
        counter += 1
print 'Train Accuracy: %f' % (counter / float(y.size) * 100.0)
