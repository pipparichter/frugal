import pandas as pd 
import numpy as np 
from sklearn.cluster import BisectingKMeans # , OPTICS
from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
import warnings 
import torch 
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# TODO: Is there any data leakage by fitting a StandardScaler before clustering? I don't think any more so
#   than caused by clustering the training and testing dataset together. 
# TODO: Why does KMeans need to use Euclidean distance? Other distance metrics also have a concept of closeness. 
# https://www.biorxiv.org/content/10.1101/2024.11.13.623527v1.full

class Clusterer():

    def __init__(self, tolerance=1e-5, n_clusters:int=1000, n_init:int=10, max_iter:int=300):
        
        self.n_clusters = n_clusters
        self.kmeans = BisectingKMeans(n_clusters=n_clusters, bisecting_strategy='largest_cluster', tol=tolerance, n_init=n_init, random_state=42, max_iter=max_iter) # Will use Euclidean distance. 
        self.scaler = StandardScaler() # I think scaling prior to clustering is important. Applying same assumption as with Classifier training. 
        self.cluster_map = None
        self.cluster_labels = None
        self.index = None

    def _check_homogenous(self, dataset):
        df = pd.DataFrame(index=dataset.index)
        df['label'] = dataset.label 
        df['cluster_label'] = self.cluster_labels
        for cluster_label, cluster_df in df.groupby('cluster_label'):
            assert cluster_df.label.nunique() == 1, f'Clusterer._check_homogenous: Cluster {cluster_label} is not homogenous.'

    # def converged(self):
    #     return (self.kmeans.n_iter_ < self.kmeans.max_iter)
    
    @staticmethod
    def _get_embeddings(dataset):
        # Use half precision to reduce memory. 
        return dataset.embedding.clone().to(torch.float16).numpy()
    
    def get_cluster_sizes(self):
        return np.array([(self.cluster_labels == i).sum() for i in range(self.n_clusters)])
    
    def get_distance_to_cluster_center(self, dataset):
        # cluster_centers = self.cluster_centers[self.cluster_labels, :]
        assert np.all(dataset.index == self.index), 'get_distance_to_cluster_center: The Dataset index does not match the stored index.'
        embeddings = Clusterer._get_embeddings(dataset) # Use half precision to reduce memory. 
        embeddings = self.scaler.transform(embeddings)

        # dists = np.sqrt(np.sum((embeddings - cluster_centers) ** 2, axis=1))
        # return dists 
        dists = self.kmeans.transform(embeddings)
        n_errors = (np.argmin(dists, axis=1) == self.cluster_labels).sum()
        # assert n_errors == 0, f'get_distance_to_cluster_center: {n_errors} cluster labels do not match the center with the minimum distance.'
        if n_errors > 0:
            warnings.warn(f'get_distance_to_cluster_center: {n_errors} cluster labels do not match the center with the minimum distance.')
        return dists.min(axis=1)

    
    def fit(self, dataset, check_homogenous:bool=False):

        embeddings = Clusterer._get_embeddings(dataset) # Use half precision to reduce memory. 
        embeddings = self.scaler.fit_transform(embeddings)
        self.index = dataset.index

        self.kmeans.fit(embeddings)
        # if not self.converged():
        #     warnings.warn('Clusterer.fit: The clustering algorithm did not converge.')

        self.cluster_labels = self.kmeans.labels_ 
        self.cluster_map = dict(list(zip(self.index, self.cluster_labels)))
        self.cluster_centers = self.kmeans.cluster_centers_

        if check_homogenous and (hasattr(dataset, 'label')):
            self._check_homogenous(dataset)

    def to_df(self, dataset=None):
        df = dict()
        df['id'] = self.index 
        df['cluster_label'] = self.cluster_labels
        if dataset is not None:
            df['distance_to_cluster_center'] = self.get_distance_to_cluster_center(dataset)
        df = pd.DataFrame(df).set_index('id')
        return df

    def write(self, path:str, dataset=None):
        df = self.to_df(dataset=dataset)
        return df.to_csv(path)


