import numpy as np


def load_PCA_data(filename):
    
    data = np.loadtxt(filename, dtype = "float", delimiter = "\t")
    data_mat = np.mat(data)
    
    return data_mat


def pca(data_mat, k):
    
    means = np.mean(data_mat, axis=0)
    new_data_mat = data_mat - means
    cov = np.cov(new_data_mat, rowvar=0)
    eigen_val, eigen_vec = np.linalg.eig(cov)
    new_eigen_val = eigen_val.argsort()[::-1][:k]
    trun_eigen_vec = eigen_vec[:, new_eigen_val] 
    reduced_data_mat = new_data_mat * trun_eigen_vec
    
    return trun_eigen_vec.T


data_mat = load_PCA_data('pca-data.txt')
trun_eigen_vec = pca(data_mat, 2)
print('Directions of the first two principal components are: \n{}'.format(trun_eigen_vec))

