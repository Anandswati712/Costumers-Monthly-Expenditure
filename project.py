#Heirarcal Clustering
# import libraries  

from this import d
import pandas as pd
import matplotlib.pyplot as plt

#import databasee
dataset = pd.read_csv("python/Mall_Customers.csv")
x = dataset.iloc[:, :].values
print(x)

#Dendrogram to find optimal num of clusters
import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
#plt.show()

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters= 5, affinity = 'euclidean', linkage = 'ward')
y_hc = clustering.fit_predict(x)

#visualising the clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0,1], c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1,1], c = 'green', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2,1], c = 'pink', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3,1], c = 'blue', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4,1], c = 'orange', label = 'Cluster 5')
plt.title("Cluster of Customers")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score(1-100")
plt.legend()
plt.show()