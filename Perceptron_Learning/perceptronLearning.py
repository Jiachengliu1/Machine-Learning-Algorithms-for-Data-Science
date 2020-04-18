import numpy as np


def predict_label(X,w):
    
    predict = np.dot(X,w)
    predict[predict>=0] = 1
    predict[predict<0] = -1
    
    return predict


def perceptron_learning(X, y, learning_rate):
    
    N, d = X.shape
    w = np.random.random(d+1)
    ones = np.ones(N).reshape(N, 1)
    X = np.concatenate((X, ones), axis = 1)
    iteration_count = 0
    while True:
        for i in range(N):
            if np.dot(w, X[i]) < 0 and y[i] == 1:
                w += learning_rate * X[i]
            elif np.dot(w, X[i]) >= 0 and y[i] == -1:
                w -= learning_rate * X[i]
        iteration_count += 1
        predict = predict_label(X, w)
        if np.all(predict==y) or iteration_count == 7000:
            break

    return X, w, iteration_count


def calculate_accuracy(X, y, w):
    
    predict = predict_label(X, w)
    correct_instances = len(np.where(predict == y)[0])
    total_instances = X.shape[0]
    misclassified_instances = total_instances - correct_instances
    accuracy = correct_instances / total_instances

    return accuracy, misclassified_instances


data = np.loadtxt('classification.txt', dtype = "float", delimiter = ",")
X = data[:,0:3]
y = data[:,3]
X, w, iteration_count = perceptron_learning(X, y, 0.01)
print('Weight: {}'.format(w))
print('Iteration counts: {}'.format(iteration_count))
accuracy, misclassified_instances = calculate_accuracy(X, y, w)
print('Accuracy: {}'.format(accuracy))
print('Number of misclassified instances: {}'.format(misclassified_instances))

