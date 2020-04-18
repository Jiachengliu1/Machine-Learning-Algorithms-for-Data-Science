import numpy as np
import matplotlib.pyplot as plt


def predict_label(X,w):
    
    predict = np.dot(X,w)
    predict[predict>=0] = 1
    predict[predict<0] = -1
    
    return predict


def pocket(X, y, learning_rate):
    
    N, d = X.shape
    w = np.random.random(d+1)
    ones = np.ones(N).reshape(N, 1)
    X = np.concatenate((X, ones), axis = 1)
    iteration_count = 0
    best_w = w
    best_accuracy = 0
    best_iteration = 0
    best_misclassified_instances = N    
    iteration_count_li = []
    misclassified_instances_li = []
    while True:
        for i in range(N):
            if np.dot(w, X[i]) < 0 and y[i] == 1:
                w += learning_rate * X[i]
            elif np.dot(w, X[i]) >= 0 and y[i] == -1:
                w -= learning_rate * X[i]
        iteration_count += 1
        accuracy, misclassified_instances = calculate_accuracy(X, y, w)
        iteration_count_li.append(iteration_count)
        misclassified_instances_li.append(misclassified_instances)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_w = w
            best_iteration = iteration_count
            best_misclassified_instances = misclassified_instances
        predict = predict_label(X, w)
        if np.all(predict==y) or iteration_count == 7000:
            break
    
    return X, best_w, best_accuracy, best_iteration, misclassified_instances, iteration_count, iteration_count_li, misclassified_instances_li


def calculate_accuracy(X, y, w):
    
    predict = predict_label(X, w)
    correct_instances = len(np.where(predict == y)[0])
    total_instances = X.shape[0]
    misclassified_instances = total_instances - correct_instances
    accuracy = correct_instances / total_instances

    return accuracy, misclassified_instances

    
def plot(iteration_count_li, misclassified_instances_li):
    
    plt.xlabel('Number of iterations')
    plt.ylabel('Number of misclassified points')
    plt.plot(iteration_count_li, misclassified_instances_li)
    plt.show()

    
data = np.loadtxt('classification.txt', dtype = "float", delimiter = ",")
X = data[:,0:3]
y = data[:,4]
X, best_w, best_accuracy, best_iteration, best_misclassified_instances, iteration_count, iteration_count_li, misclassified_instances_li = pocket(X, y, 0.01)
print('Best weight: {}'.format(best_w))
print('Iteration counts: {}'.format(iteration_count))
print('Best accuracy: {}'.format(best_accuracy))
print('Best iteration: {}'.format(best_iteration))
print('Number of misclassified instances: {}'.format(best_misclassified_instances))
plot(iteration_count_li, misclassified_instances_li)

