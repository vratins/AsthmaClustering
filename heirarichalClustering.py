from sklearn.datasets import make_circles
from sklearn.cluster import AgglomerativeClustering
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as shc


# Notes:
# Rows - Samples
# Columns - Genes 

def averageLinkageSK(data):
    # Example with Random Data Points - Testing
    # data, _ = make_circles(n_samples=1000, noise=0.05, random_state=0)
    # ward = AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="ward")
    # ward.fit(data)
    # ward_labels = ward.labels_
    # plt.scatter(data[:, 0], data[:, 1],c=ward_labels, cmap="Paired")
    # plt.title("Ward")
    # plt.show()
    # print("Done!")

    data = data.T
    # clusters = shc.linkage(data, method='ward', metric="euclidean")
    # plt.figure(figsize=(10, 7))
    # plt.title("Asthma Dendrogram")
    # shc.dendrogram(Z=clusters)
    # plt.show()
    print(data)
    print("Columns: ", data.columns)
    clustering_model = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    clustering_model.fit(data)
    clustering_model.labels_
    data_labels = clustering_model.labels_
    sns.scatterplot(x='0', 
                    y='1', 
                    data=data, 
                    hue=data_labels,
                    palette="rainbow").set_title('Labeled Customer Data')


def averageLinkage (data):
    print("Average Linkage Implementation of Heirarichal Clustering without SciKit")

    numClust = 4 #NC, notSA, SA, VSA
    data = data.T # COL = genes, ROW = samples



small_data = pd.read_csv("Data/BAL/TestBALData.tsv", sep = '\t', usecols=range(2,6))
total_data = pd.read_csv("Data/BAL/normdataBAL0715.txt", sep = '\t', usecols=range(2,6))
averageLinkageSK(small_data)