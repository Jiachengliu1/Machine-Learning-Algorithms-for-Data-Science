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


def random_centroids(data, k):
    
    centroids = np.mat(np.zeros((k, data.shape[1])))
    centroid = random.sample(range(data.shape[0]), 3)
    centroids = data[centroid]    
    
    return centroids


def distance(vector1, vector2):
    
    distance = np.sqrt(np.sum(np.power(vector1 - vector2, 2))) 
    
    return distance


def k_means(data, centroids):
    
    iter0 = 0
    convergence = False
    new_centroids = np.mat(np.zeros((centroids.shape[0], 2)))
    while not convergence:
        iter0 += 1
        point_cluster = np.mat(np.zeros((data.shape[0], 1)))
        for i in range(data.shape[0]):
            min_dist = np.inf
            cluster_num = -1
            for k in range(centroids.shape[0]):
                dist = distance(centroids[k], data[i])
                if dist < min_dist:
                    min_dist = dist,
                    cluster_num = k
            point_cluster[i] = cluster_num
        for i in range(centroids.shape[0]):
            all_points = data[np.nonzero(point_cluster == i)[0]]
            new_centroids[i] = np.mean(all_points, axis = 0)
        #print(new_centroids)
        if (np.abs(new_centroids - centroids) < 0.01).all() or iter0 > 100:
            convergence = True
        else:
            centroids = new_centroids.copy()
            
    return(new_centroids, iter0)


data = np.mat(load_data('clusters.txt'))
centroids = random_centroids(data, 3)
new_centroids, iteration = k_means(data, centroids)
print('Centroids of K-means algorithm is: \n', new_centroids)
print('Iteration counts:', iteration)

