import numpy as np
from cvxopt import matrix, solvers


class linearSVM:

    def __init__(self, X, y):

        self.X = X
        self.y = y
        self.M, self.N = X.shape
        self.kernel = np.zeros((self.M, self.M))
        
        for i in range(self.M):
            for j in range(self.M):
                self.kernel[i, j] = np.dot(self.X[i], self.X[j])
                
        self.P = matrix(np.outer(self.y,self.y) * self.kernel)
        self.q = matrix(np.negative(np.ones(self.M)))
        self.A = matrix(self.y, (1,self.M))
        self.b = matrix(0.0)
        self.G = matrix(np.negative(np.identity(self.M)))
        self.h = matrix(np.zeros(self.M))
        
    def qp(self):
        
        solver = solvers.qp(self.P, self.q, self.G, self.h, self.A, self.b)
        self.alphas = np.array(solver['x']).reshape(1,self.M)[0]
        
        return self.alphas
    
    def fit(self):

        self.alphas = self.qp()       
        self.support_vector_indices = np.where(self.alphas > 0.00001)[0]
        # weights
        self.weights = np.zeros((1, self.N))[0]
        for i in np.nditer(self.support_vector_indices):
            self.weights += np.array(self.alphas[i] * y[i] * X[i])
        # support vectors
        self.support_vectors = self.X[self.support_vector_indices]
        # bias
        index = self.support_vector_indices[0]
        self.bias = y[index] - np.dot(self.weights, X[index])
        slope = - self.weights[0] / self.weights[1]
        intercept = - self.bias / self.weights[1]
        print('Support vectors: {}'.format(self.support_vectors))        
        print('Equation: y = {}x + {}'.format(slope, intercept))
        
    def predict(self, X_test):
        
        label = int(round(np.dot(self.weights, X_test) + self.bias))
        if np.dot(self.weights, X_test) + self.bias >= 1:
            label = 1
        elif np.dot(self.weights, X_test) + self.bias <= -1:
            label = -1
        
        return label
            
            
if __name__ == '__main__':
    
    data = np.loadtxt('linsep.txt',dtype='float',delimiter=',')
    X = data[:,0:2]
    y = data[:,2]
    linear_svm = linearSVM(X, y)
    linear_svm.fit()
    predictions = []
    for i in range(len(X)):
        label = linear_svm.predict(X[i])
        predictions.append(label)
    print('Accuracy: {}'.format(np.sum(predictions == y) / len(y)))
    
    