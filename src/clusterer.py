import pandas as pd 
import numpy as np 
from sklearn.cluster import BisectingKMeans # , OPTICS
from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
import warnings 
import torch 
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances

# TODO: Is there any data leakage by fitting a StandardScaler before clustering? I don't think any more so
#   than caused by clustering the training and testing dataset together. 
# TODO: Why does KMeans need to use Euclidean distance? Other distance metrics also have a concept of closeness. 
# https://www.biorxiv.org/content/10.1101/2024.11.13.623527v1.full

class Clusterer():

    def __init__(self, tolerance=1e-3, n_clusters:int=1000, n_init:int=3, max_iter:int=50):
        
        self.n_clusters = n_clusters
        self.kmeans = BisectingKMeans(n_clusters=n_clusters, bisecting_strategy='largest_cluster', tol=tolerance, n_init=n_init, random_state=42, max_iter=max_iter) # Will use Euclidean distance. 
        self.scaler = StandardScaler() # I think scaling prior to clustering is important. Applying same assumption as with Classifier training. 
        self.cluster_map = None
        self.cluster_labels = None

    def _check_homogenous(self, dataset):
        df = pd.DataFrame(index=dataset.index)
        df['label'] = dataset.label 
        df['cluster_label'] = self.cluster_labels
        for cluster_label, cluster_df in df.groupby('cluster_label'):
            assert cluster_df.label.nunique() == 1, f'Clusterer._check_homogenous: Cluster {cluster_label} is not homogenous.'

    def converged(self):
        return (self.kmeans.n_iter_ < self.kmeans.max_iter)
    
    def get_cluster_sizes(self):
        return np.array([(self.cluster_labels == i).sum() for i in range(self.n_clusters)])
    
    def get_cluster_pairwise_distances(self, dataset, cluster_label):
        embeddings = dataset.embedding.to(torch.float16) # Use half precision to reduce memory. 
        embeddings = embeddings[self.cluster_labels == cluster_label]
        embeddings = self.scaler.transform(embeddings)



    def fit(self, dataset, check_homogenous:bool=True, get_diameters:bool=True):

        embeddings = dataset.embedding.to(torch.float16) # Use half precision to reduce memory. 
        embeddings = self.scaler.fit_transform(embeddings)
        index = dataset.index

        self.kmeans.fit(embeddings)
        if not self.converged():
            warnings.warn('Clusterer.fit: The clustering algorithm did not converge.')

        self.cluster_labels = self.kmeans.labels_ 
        self.cluster_map = list(dict(zip(index, self.cluster_labels)))
        self.cluster_centers = self.kmeans.cluster_centers_

        if check_homogenous and (hasattr(dataset, 'label')):
            self._check_homogenous(dataset)


    def write(self, path:str):
        df = pd.DataFrame.from_dict(self.cluster_map, orient='index', columns=['cluster_label'])
        df.index.name = 'id'
        if self.diameters is not None:
            df['cluster_diameter'] = df.cluster_label.map(self.diameters)

        df.to_csv(path)

