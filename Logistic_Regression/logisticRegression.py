import numpy as np


def sigmoid(s):
    
    return np.exp(s) / (1 + np.exp(s))


def calculate_weight(X, y, w, learning_rate):
    
    N, d = X.shape
    y = y.reshape(1, N)
    for i in range(7000):
        s = np.dot(y, np.dot(X, w))
        base = sigmoid(-s) * np.dot(y, X)
        base = np.sum(base)
        base = - base / N
        w -= learning_rate * base
        
    return w


def predict_label(X,w):
    
    predict = np.dot(X,w)
    predict[predict>=0] = 1
    predict[predict<0] = -1
    
    return predict


def calculate_accuracy(X, y, w):
    
    predict = predict_label(X, w)
    correct_instances = len(np.where(predict == y)[0])
    total_instances = X.shape[0]
    accuracy = correct_instances / total_instances

    return accuracy


data = np.loadtxt('classification.txt', dtype = "float", delimiter = ",")
X = data[:,0:3]
N, d = X.shape
ones = np.ones(N).reshape(N, 1)
X = np.concatenate((X, ones), axis = 1)
y = data[:,4]
w = np.random.random(d+1)
w = calculate_weight(X, y, w, 0.01)
print('Weight: {}'.format(w))
accuracy = calculate_accuracy(X, y, w)
print('Accuracy: {}'.format(accuracy))

