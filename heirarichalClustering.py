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
from scipy.cluster.hierarchy import linkage, dendrogram

def SKheirarichalClustering(data, clusterNum):
    clusters = AgglomerativeClustering(n_clusters=clusterNum, linkage='average')
    clusterLabels = clusters.fit(data).labels_
    return clusters, clusterLabels

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
    # print(dist_matrix)
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
    # print(clusters)
    return clusters

# MAIN

# different datas lol choose ur fav at this point, jk be smart
small_data = pd.read_csv("Data/BAL/TestBALData.tsv", sep = '\t', usecols=range(2,6)).T
total_data = pd.read_csv("Data/BAL/normdataBAL0715.txt", sep = '\t', usecols=range(2,6)).T
filtered_data = pd.read_csv("filtered_data_BAL.csv", sep = ',').T
sample_data = (datasets.load_iris()).data #default data set from sklearn for testing

clusterNum = 4
data = filtered_data # change based on data

# Self Implementation of Heirarichal Clustering 
clusters = heirarichalClustering(data, clusterNum)
distMatrix = distanceMatrix(data)
Z = linkage(distMatrix, method='average')
plt.figure(figsize=(10, 5))
dendrogram(Z, labels=['Sample {}'.format(i) for i in range(data.shape[0])])
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Dendrogram of Implemented Heirarichal Clustering')
plt.show()
# plt.figure(figsize=(8, 6))
# for i, cluster in enumerate(clusters):
#     plt.scatter(data[cluster, 0], data[cluster, 1], label=f'Cluster {i+1}')
# plt.legend()
# plt.title('Hierarchical Clustering with Average Linkage w/ Own Implementation')
# plt.xlabel('50')
# plt.ylabel('144')
# plt.show()

# Scikit-learn's implementation of Heirarichal Clustering 
SKclusters, SKclusterLabels = SKheirarichalClustering(data, clusterNum)
# assignedClusters = []
# for i in range(clusterNum):
#     indivCluster = []
#     for j in range(len(SKclusters)):
#         if SKclusters[j] == i:
#             indivCluster.append(j)
#     assignedClusters.append(indivCluster)
SKclusters.fit(data)
Z = linkage(SKclusters.children_, method='average')
plt.figure(figsize=(10, 8))
dendrogram(Z)
plt.title('Dendrogram of SciKits Hierarchical Clustering')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()



# plt.figure(figsize=(8, 6))
# for i in range(clusterNum):
#     plt.scatter(data[SKclusters == i, 0], data[SKclusters == i, 1], label=f'Scikit Cluster {i+1}')
# plt.legend()
# plt.title('Hierarchical Clustering with Average Linkage w/ SciKits Implementation')
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.show()