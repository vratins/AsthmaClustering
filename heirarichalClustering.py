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


# Notes:
# Rows - Samples
# Columns - Genes 

def averageLinkageSK(data):
    iris = datasets.load_iris()
    data=pd.DataFrame(iris['data'])
    distance_matrix = linkage(data, method = 'ward', metric = 'euclidean')
    data['cluster_labels'] = fcluster(distance_matrix, 3, criterion='maxclust')
    data['target'] = iris.target
    fig, axes = plt.subplots(1, 2, figsize=(16,8))
    axes[0].scatter(data[0], data[1], c=data['target'])
    axes[1].scatter(data[0], data[1], c=data['cluster_labels'], cmap=plt.cm.Set1)
    axes[0].set_title('Actual', fontsize=18)
    axes[1].set_title('Hierarchical', fontsize=18)
    plt.show()

    # # Example with Random Data Points - Testing
    # # data, _ = make_circles(n_samples=1000, noise=0.05, random_state=0)
    # # ward = AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="ward")
    # # ward.fit(data)
    # # ward_labels = ward.labels_
    # # plt.scatter(data[:, 0], data[:, 1],c=ward_labels, cmap="Paired")
    # # plt.title("Ward")
    # # plt.show()
    # # print("Done!")

    # data = data.T
    # # clusters = shc.linkage(data, method='ward', metric="euclidean")
    # # plt.figure(figsize=(10, 7))
    # # plt.title("Asthma Dendrogram")
    # # shc.dendrogram(Z=clusters)
    # # plt.show()
    # print(data)
    # print("Columns: ", data.columns)
    # clustering_model = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    # clustering_model.fit(data)
    # clustering_model.labels_
    # data_labels = clustering_model.labels_
    # sns.scatterplot(x='0', 
    #                 y='1', 
    #                 data=data, 
    #                 hue=data_labels,
    #                 palette="rainbow").set_title('Labeled Customer Data')


small_data = pd.read_csv("Data/BAL/TestBALData.tsv", sep = '\t', usecols=range(2,6))
total_data = pd.read_csv("Data/BAL/normdataBAL0715.txt", sep = '\t', usecols=range(2,6))

sample = datasets.load_iris()
sample_data = sample.data
#averageLinkage Running Stuff
# d, target = averageLinkage(_data)
# print("Target:")
# print(target)
# target[target == 140] = 0
# target[target == 144] = 1
# target[target == 146] = 2
# plt.scatter(iris_data[:,1], iris_data[:,2], c=target, cmap="rainbow")
# plt.show()

def euclideanDistance(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2)))

def distanceMatrix (data):
    # length = data.shape[0]
    # A_dots = (data*data).sum(axis=1).reshape((length,1))*np.ones(shape=(1,length))
    # B_dots = (data*data).sum(axis=1)*np.ones(shape=(length,1))
    # matrix =  A_dots + B_dots -2*data.dot(data.T)
    # zero_mask = np.less(matrix, 0.0)
    # matrix[zero_mask] = 0.0
    # print(np.sqrt(matrix))
    n = data.shape[0]
    distMatrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distMatrix[i,j] = euclideanDistance(data[i], data[j])
            distMatrix[j,i] = distMatrix[i,j]
    print(distMatrix)
    return distMatrix

def heirarichalClustering (data, clusterNum):
    # Number of samples
    n = data.shape[0]

    # Creates distance matrix
    dist_matrix = distanceMatrix(data)

    # Places each data point into it's own cluster
    clusters = [[i] for i in range(n)]

    # Repeats until you've reached to a certain number of clusters, condensed from number of sample clsuters to number of clsuters needed
    while len(clusters) > clusterNum:
        # Find the indices of the two closest clusters
        min_dist = float('inf')
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = np.mean([dist_matrix[a,b] for a in clusters[i] for b in clusters[j]])
                if dist < min_dist:
                    min_dist = dist
                    merge_indices = (i, j)
        # Merge the two closest clusters
        clusters[merge_indices[0]] += clusters[merge_indices[1]]
        del clusters[merge_indices[1]]
    print(clusters)

# MAIN
sample = datasets.load_iris()
sample_data = sample.data
clusterNum = 4
heirarichalClustering(sample_data, clusterNum)
