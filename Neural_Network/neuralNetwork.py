import numpy as np
import cv2


def load_data(filename):
    
    X = []
    y = []
    for row in open(filename):
        if 'down' in row:
            down_gesture = cv2.imread(row.strip(), -1)
            M, N = down_gesture.shape
            down_gesture = list(down_gesture.reshape(M*N))
            X.append(down_gesture)
            y.append(1)
        else:
            other_gesture = cv2.imread(row.strip(), -1)
            M, N = other_gesture.shape
            other_gesture = list(other_gesture.reshape(M*N))
            X.append(other_gesture)
            y.append(0)
    X = np.asarray(X)  
    y = np.asarray(y)
    
    return X, y


class NN:
    
    def __init__(self, layers):

        self.w = []
        for i in range(1, len(layers) - 1):
            self.w.append((np.random.random((layers[i - 1], layers[i]))-1)*0.01)
            self.w.append((np.random.random((layers[i], layers[i + 1]))-1)*0.01)

    def sigmoid(self, s):
        
        return 1/(1 + np.exp(-s))

    def fit(self, X, y, learning_rate, epochs):
 
        for i in range(epochs):
            j = np.random.randint(X.shape[0])
            x = [X[j]]
            for i in range(len(self.w)): 
                x.append(self.sigmoid(np.dot(x[i], self.w[i]))) 
            # base case
            deltas = [2*(x[-1] - y[j]) * np.multiply(self.sigmoid(x[-1]),(1-self.sigmoid(x[-1])))]
            # back propagation
            for i in range(len(x) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.w[i].T)*np.multiply(self.sigmoid(x[i]),(1-self.sigmoid(x[i])))) 
            deltas.reverse()
            for i in range(len(self.w)):
                layer = np.atleast_2d(x[i])
                delta = np.atleast_2d(deltas[i])
                self.w[i] -= learning_rate * layer.T.dot(delta)
 
    def predict(self, x):
        
        y = x
        for i in range(len(self.w)):
            y = self.sigmoid(np.dot(y, self.w[i]))
        if float(y) >= 0.5:
            y = 1
        else:
            y = 0

        return y

    
X_train, y_train = load_data('downgesture_train.list')
X_test, y_test = load_data('downgesture_test.list')
nn = NN([960, 100, 1])
nn.fit(X_train, y_train, 0.1, 1000)
predict_result = []
for x in X_test:
    predict_result.append(nn.predict(x))
predict_result = np.asarray(predict_result)
print('Predictions: {}'.format(predict_result))
print('Accuracy: {}'.format(np.sum(predict_result == y_test) / len(y_test)))

