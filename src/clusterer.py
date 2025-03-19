import pandas as pd 
import numpy as np 
from sklearn.cluster import DBSCAN # , OPTICS
from sklearn.neighbors import NearestNeighbors
import torch 
from scipy.spatial import distance_matrix


class Clusterer():

    def __init__(self, radius:float=None, min_samples:int=3):
        
        self.dbscan = DBSCAN(metric='precomputed', min_samples=min_samples) 
        # self.nearest_neighbors = None
        self.radius = radius 
        self.clusters = None
        self.n_singleton_clusters = None
        self.cluster_map = None
        self.diameters = None

    def _fit_radius(self, dataset):
        embeddings = dataset.embedding.to(torch.float16) # Use half precision to reduce memory. 
        n_dists = (dataset.label == 0).sum() * (dataset.label == 1).sum()
        memory = 2 * n_dists * 1e-9
        print(f'Clusterer._fit_radius: Computing {n_dists} distances, requiring {memory:.3f} GB.')

        dists = torch.cdist(embeddings[dataset.label == 0], embeddings[dataset.label == 1], p=2) # Compute distances between real and spurious points. 
        radius = min(dists.ravel()).item() # The minimum distance between a real and spurious point. 
        
        print(f'Clusterer._fit_radius: Clustering using neighborhood radius {radius}.')
        self.radius = radius
    
    @staticmethod
    def _check_homogenous(dataset, cluster_labels:np.ndarray):

        df = pd.DataFrame(index=dataset.index)
        df['label'] = dataset.label 
        df['cluster_label'] = cluster_labels

        for cluster_label, cluster_df in df.groupby('cluster_label'):
            assert cluster_df.label.nunique() == 1, f'Clusterer._check_homogenous: Cluster {cluster_label} is not homogenous.'

    def _get_diameters(self, dataset):
        df = pd.DataFrame(dataset.embedding, index=dataset.index)
        df['cluster_label'] = df.index.map(self.cluster_map)

        self.diameters = dict()
        for cluster_label, cluster_df in df.groupby('cluster_label'):
            if len(cluster_df) == 1:
                self.diameters[cluster_label] = 0
                continue

            dists = distance_matrix(cluster_df.values, cluster_df.values, p=2)
            self.diameters[cluster_label] = dists.max().item()

    def fit(self, dataset, check_homogenous:bool=True, get_diameters:bool=True):

        embeddings = dataset.embedding.to(torch.float16) # Use half precision to reduce memory. 
        index = dataset.index

        if self.radius is None:
            self._fit_radius(dataset) 

        nearest_neighbors = NearestNeighbors(metric='minkowski', p=2, radius=self.radius)
        nearest_neighbors.fit(embeddings)
        self.graph = nearest_neighbors.radius_neighbors_graph(X=embeddings, radius=self.radius, mode='distance', sort_results=True)

        self.dbscan.fit(self.graph)
        # Samples that DBSCAN considers "noisy," i.e. can't be assigned a cluster, are given labels -1. 
        # These should be put in their own clusters. 
        cluster_labels = self.dbscan.labels_

        max_cluster_label = max(cluster_labels)
        n_outliers = (cluster_labels < 0).sum()
        cluster_labels[cluster_labels < 0] = np.arange(max_cluster_label + 1, max_cluster_label + n_outliers + 1) # Assign cluster labels to the outliers. 

        self.n_clusters = len(np.unique(cluster_labels))
        self.cluster_map = dict(list(zip(index, cluster_labels)))
        self.clusters = {i:list(index[cluster_labels == i]) for i in range(self.n_clusters)}
        self.n_singleton_clusters = n_outliers

        if check_homogenous:
            Clusterer._check_homogenous(dataset, cluster_labels)
        if get_diameters:
            self._get_diameters(dataset)

    def write(self, path:str):
        df = pd.DataFrame.from_dict(self.cluster_map, orient='index', columns=['cluster_label'])
        df.index.name = 'id'
        if self.diameters is not None:
            df['cluster_diameter'] = df.cluster_label.map(self.diameters)

        df.to_csv(path)

