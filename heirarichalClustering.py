from sklearn.datasets import make_circles
from sklearn.cluster import AgglomerativeClustering
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as shc
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

def SKheirarichalClustering(data):
    clusters = AgglomerativeClustering(n_clusters=4, linkage='average').fit(data).labels_
    return clusters

def euclideanDistance(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2)))

def distanceMatrix (data):
    n = data.shape[0]
    distMatrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distMatrix[i,j] = euclideanDistance(data[i], data[j])
            distMatrix[j,i] = distMatrix[i,j]
    return distMatrix

def heirarichalClustering (data, clusterNum):
    # Number of samples
    n = data.shape[0]
    # Creates distance matrix
    dist_matrix = distanceMatrix(data)
    
    # Places each data point into it's own cluster
    clusters = [[i] for i in range(n)]
    # Repeats until you have reduced from #sample clusters to #specified clusters
    while len(clusters) > clusterNum:
        # Finds closest clusters
        min = 1000000
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = np.mean([dist_matrix[a,b] for a in clusters[i] for b in clusters[j]])
                if dist < min:
                    min = dist
                    merged = (i, j)

        # Merge the two closest clusters
        clusters[merged[0]] += clusters[merged[1]]
        del clusters[merged[1]]
    return clusters

# MAIN
small_data = pd.read_csv("Data/BAL/TestBALData.tsv", sep = '\t', usecols=range(2,6))
total_data = pd.read_csv("Data/BAL/normdataBAL0715.txt", sep = '\t', usecols=range(2,6))

sample = datasets.load_iris()
sample_data = sample.data
clusterNum = 4

# Self Implementation of Heirarichal Clustering 
clusters = heirarichalClustering(sample_data, clusterNum)
plt.figure(figsize=(8, 6))
for i, cluster in enumerate(clusters):
    plt.scatter(sample_data[cluster, 0], sample_data[cluster, 1], label=f'Cluster {i+1}')
plt.legend()
plt.title('Hierarchical Clustering with Average Linkage w/ Own Implementation')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

# Scikit-learn's implementation of Heirarichal Clustering 
SKclusters = SKheirarichalClustering(sample_data)
plt.figure(figsize=(8, 6))
for i in range(clusterNum):
    plt.scatter(sample_data[SKclusters == i, 0], sample_data[SKclusters == i, 1], label=f'Scikit Cluster {i+1}')
plt.legend()
plt.title('Hierarchical Clustering with Average Linkage w/ SciKits Implementation')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()