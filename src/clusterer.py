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
import pickle

# TODO: Is there any data leakage by fitting a StandardScaler before clustering? I don't think any more so
#   than caused by clustering the training and testing dataset together. 
# TODO: Why does KMeans need to use Euclidean distance? Other distance metrics also have a concept of closeness. 
# https://www.biorxiv.org/content/10.1101/2024.11.13.623527v1.full

class Clusterer():

    def __init__(self, tolerance=1e-8, n_clusters:int=1000, n_init:int=10, max_iter:int=1000, verbose:bool=False):
        
        self.n_clusters = n_clusters
        self.kmeans = BisectingKMeans(verbose=verbose, n_clusters=n_clusters, bisecting_strategy='largest_cluster', tol=tolerance, n_init=n_init, random_state=42, max_iter=max_iter) # Will use Euclidean distance. 
        self.scaler = StandardScaler() # I think scaling prior to clustering is important. Applying same assumption as with Classifier training. 
        self.cluster_map = None
        self.cluster_labels = None
        self.index = None # Stores the index of the data used to fit the model. 

    def _check_homogenous(self, dataset):
        df = pd.DataFrame(index=dataset.index)
        df['label'] = dataset.label 
        df['cluster_label'] = self.cluster_labels
        for cluster_label, cluster_df in df.groupby('cluster_label'):
            assert cluster_df.label.nunique() == 1, f'Clusterer._check_homogenous: Cluster {cluster_label} is not homogenous.'

    def transform(self, dataset):
        embeddings = dataset.numpy().astype(np.float16) # Use half precision to reduce memory. 
        embeddings = self.scaler.transform(embeddings)
        dists = self.kmeans.transform(embeddings)
        return dists 
    
    def predict(self, dataset):
        dists = self.transform(dataset)
        return dists.argmin(axis=1) 
    
    def fit(self, dataset): # , check_homogenous:bool=False):
        embeddings = dataset.numpy().astype(np.float16) # Use half precision to reduce memory. 
        embeddings = self.scaler.fit_transform(embeddings)
        self.index = dataset.index
        self.kmeans.fit(embeddings)
        self.cluster_labels = self.kmeans.labels_ 
        self.cluster_map = dict(list(zip(self.index, self.cluster_labels)))
        self.cluster_centers = self.kmeans.cluster_centers_

        # if check_homogenous and (hasattr(dataset, 'label')):
        #     self._check_homogenous(dataset)
    
    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path:str):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj





