import numpy as np
from cvxopt import matrix, solvers


class nonlinearSVM:

    def __init__(self, X, y):

        self.X = X
        self.y = y
        self.M, self.N = X.shape
        self.kernel = np.zeros((self.M, self.M))
        
        for i in range(self.M):
            for j in range(self.M):
                self.kernel[i, j] = (1 + np.dot(self.X[i], self.X[j])) ** 2
                
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
        self.alphas = self.alphas[self.support_vector_indices][:, np.newaxis]
        # support vectors
        self.support_vectors = self.X[self.support_vector_indices]
        # bias  
        m = len(self.support_vector_indices)
        self.z = np.zeros((m, 1))
        for i in range(m):
            self.z[i] = self.kernel[self.support_vector_indices[i], self.support_vector_indices[0]]
        self.bias = self.y[self.support_vector_indices[0]] - np.sum(self.alphas * self.y[self.support_vector_indices][:, np.newaxis] * self.z, axis=0)
        print('Support vectors: {}'.format(self.support_vectors))           
        
    def predict(self, X_test):
        
        item = 0
        for alpha, sv_x, sv_y in zip(self.alphas, self.support_vectors, self.y[self.support_vector_indices]):
            item += alpha * sv_y * ((1 + np.dot(X_test, sv_x)) ** 2)
        y_test = (item + self.bias)[0]
        label = 0
        if y_test > 0:
            label = 1
        elif y_test < 0:
            label = -1
        
        return label
    
    
if __name__ == '__main__':
    
    data = np.loadtxt('nonlinsep.txt',dtype='float',delimiter=',')
    X = data[:,0:2]
    y = data[:,2] 
    nonlinear_svm = nonlinearSVM(X, y)
    nonlinear_svm.fit()
    predictions = []
    for i in range(len(X)):
        label = nonlinear_svm.predict(X[i])
        predictions.append(label)
    print('Accuracy: {}'.format(np.sum(predictions == y) / len(y)))

    