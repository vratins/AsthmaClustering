import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

def pca(X, D):
    """
    PCA
    Input:
        X - An (1024, 1024) numpy array
        D - An int less than 1024. The target dimension
    Output:
        X - A compressed (1024, 1024) numpy array
    """
    u, e, v = np.linalg.svd(np.dot(X.T, X))
    eigen = v.T[:, 0:D]
    final = np.dot(X, eigen)
    #return np.dot(final, eigen.T)
    return final


def sklearn_pca(X, D):
    """
    Your PCA implementation should be equivalent to this function.
    Do not use this function in your implementation!
    """
    from sklearn.decomposition import PCA
    p = PCA(n_components=D, svd_solver='full')
    trans_pca = p.fit_transform(X)
    X = p.inverse_transform(trans_pca)
    return X


if __name__ == '__main__':
    X = pd.read_csv("Data/BAL/normdataBAL0715.txt", sep = '\t', skiprows= range(1001, 45015))
    print(X)
    D = 2




