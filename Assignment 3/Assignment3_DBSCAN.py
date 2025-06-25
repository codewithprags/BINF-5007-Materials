"""Apply DBSCAN, k-Means, and Hierarchical Clustering to at 
least two different datasets and present results using clearly labeled plots:

● Dataset where DBSCAN excels (e.g., non-spherical clusters, datasets with noise). 
● Dataset where DBSCAN struggles (e.g., varying densities, diffi culty tuning eps).


Analysis
● Compare performance: When does DBSCAN outperform k-Means and Hierarchical Clustering?
● Discuss failure cases: When does DBSCAN struggle, and why?
● Trade-off s: What factors infl uence the choice between these clustering methods?

"""

"""Import Libraries"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors






def parameter_optimize(dataset,dataname = "Moons Dataset", min_samples_range=range(1, 5), n_clusters_range=range(1, 11) ):
    X = dataset["x"]
    Y = dataset["y"]

    figure, axis = plt.subplots(1, 3, figsize=(20, 10))
    wcss = []

    # KMeans Elbow Method
    for i in n_clusters_range:
        kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10)
        kmeans.fit(dataset)
        wcss.append(kmeans.inertia_)
    axis[0].plot(list(n_clusters_range), wcss)
    axis[0].set_title(f"Elbow Method - {dataname}")
    axis[0].set_xlabel("Number of clusters")
    axis[0].set_ylabel("WCSS")

    # k-distance plot for DBSCAN
    all_k_distances = []
    for i in min_samples_range:
        neighbors = NearestNeighbors(n_neighbors=i)
        neighbors_fit = neighbors.fit(dataset)
        distances, indices = neighbors_fit.kneighbors(dataset)
        k_distances = np.sort(distances[:, i-1])
        all_k_distances.append(k_distances)
        axis[1].plot(k_distances, label=f'min_samples={i}')
    axis[1].set_title(f"k-distance plot for - {dataname}")
    axis[1].set_xlabel("Points sorted by distance")
    axis[1].set_ylabel("k-NN distance")
    axis[1].legend()

    # Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='complete', metric='euclidean')
    y_agg = agg_clustering.fit_predict(dataset)
    axis[2].scatter(X, Y, c=y_agg, cmap='viridis', s=10)
    axis[2].set_title(f"Agglomerative Clustering- {dataname}")
    axis[2].set_xlabel("x")
    axis[2].set_ylabel("y")

    plt.show()
    return wcss, all_k_distances, n_clusters_range



    





