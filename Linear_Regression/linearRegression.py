import numpy as np


data = np.loadtxt('linear-regression.txt', dtype = "float", delimiter = ",")
X = np.matrix(data[:,0:2])
N = X.shape[0]
ones = np.ones(N).reshape(1,N)
D = np.concatenate((ones, X.T), axis = 0)
y = np.matrix(data[:,2])
y = y.T
w = np.linalg.inv(D * D.T) * D * y
print('Weight: {}'.format(w))

