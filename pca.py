import numpy as np
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
    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    u, e, v = np.linalg.svd(np.dot(X.T, X))
    eigen = v.T[:, 0:D]
    final = np.dot(X, eigen)
    return final


def sklearn_pca(X, D):
    """
    Your PCA implementation should be equivalent to this function.
    Do not use this function in your implementation!
    """
    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    from sklearn.decomposition import PCA
    p = PCA(n_components=D, svd_solver='full')
    trans_pca = p.fit_transform(X)
    return trans_pca


if __name__ == '__main__':
    X = pd.read_csv("Data/BAL/normdataBAL0715.txt", sep = '\t', usecols=range(2,154))
    D = 2
    pca_final=pca(X, D)
    pca_sklearn=sklearn_pca(X, D)

    plt.scatter(pca_final[:, 0], pca_final[:, 1])
    plt.title('PCA')
    plt.show()




