import numpy as np
import random


def load_data(filename):
    
    data = []
    f = open(filename)
    for row in f:
        row = row.strip('\n').split(',')
        row = [float(element) for element in row]
        data.append(row)
    
    return data


def Gaussian(data, mean, cov):
    
    d = np.shape(cov)[0]   
    cov_determinant = np.linalg.det(cov)
    cov_inverse = np.linalg.inv(cov) 
    x = - 1/2 * np.dot(np.dot((data - mean).T, cov_inverse), (data - mean))
    base = 1/(np.power(2 * np.pi, d/2))
    normal_distribution = base * np.power(cov_determinant, -0.5) * np.exp(x)
    
    return normal_distribution
       

def E_step(data, K, mean, cov, amp):
    
    N = data.shape[1]
    E_weight = np.zeros(150*3).reshape(150, 3)
    
    for i in range(N):
        top = [amp[c] * Gaussian(data[:,i], mean[c], cov[c]) for c in range(K)]
        bottom = np.sum(top)    
        for c in range(K):
            E_weight[i][c] = top[c] / bottom
                                                        
    return E_weight
       
    
def M_step(data, K, weight):
        
    mean = []
    cov = []
    amp = []
    N = data.shape[1]
    
    for c in range(K):
        Nc = np.sum(weight[i][c] for i in range(N))
        mean.append((1.0 / Nc) * np.sum([weight[i][c] * data[:,i] for i in range(N)], axis = 0))
        cov.append((1.0 / Nc) * np.sum([weight[i][c] * (data[:,i] - mean[c]).reshape(2,1) * (data[:,i] - mean[c]).reshape(2,1).T for i in range(N)], axis = 0))
        amp.append(Nc / N)

    return mean, cov, amp


def GMM(data, K):
        
    N = data.shape[1]
    d = data.shape[0]
    weight = np.random.rand(N, K)
    for i in range(N):
        random1 = random.random()
        random2 = random.random()
        random3 = random.random()
        random_sum = random1 + random2 + random3
        weight[i] = [random1/random_sum, random2/random_sum, random3/random_sum] 
        
    cur_weight = weight
    iteration = 0
    while True:
        mean, cov, amp = M_step(data, K, cur_weight)
        new_weight = E_step(data, K, mean, cov, amp)
        if (np.abs(new_weight-cur_weight) < 0.000001).all() or iteration > 1000:
            break
        
        iteration += 1
        cur_weight = new_weight.copy()
        
    return mean, cov, amp, iteration


data = np.mat(load_data('clusters.txt'))
data = data.T
results = GMM(data, 3)
print('Results of self-implemented Gaussian Mixture Model are:')
print('Mean:', results[0])
print('Covariance:', results[1])
print('Amplitude:', results[2])
print('Iteration counts:', results[3])

    