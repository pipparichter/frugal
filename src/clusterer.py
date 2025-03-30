import pandas as pd 
import numpy as np 
from sklearn.cluster import BisectingKMeans, KMeans # , OPTICS
from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
import warnings 
import torch 
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import pickle
import re 

# TODO: Is there any data leakage by fitting a StandardScaler before clustering? I don't think any more so
#   than caused by clustering the training and testing dataset together. 
# TODO: Why does KMeans need to use Euclidean distance? Other distance metrics also have a concept of closeness. 
# https://www.biorxiv.org/content/10.1101/2024.11.13.623527v1.full



class Clusterer():

    def __init__(self, verbose:bool=True, n_clusters:int=1000, n_init:int=10, max_iter:int=1000, bisecting_strategy:str='largest_non_homogenous'):
        
        self.n_clusters = n_clusters 
        self.scaler = StandardScaler() # I think scaling prior to clustering is important. Applying same assumption as with Classifier training. 
        self.index = None # Stores the index of the data used to fit the model. 
        
        self.curr_cluster_label = 1
        self.tree = '0'
        self.split_order = []
        self.cluster_labels = None
        self.labels = None
        self.cluster_centers = [None] # Store as a list, becaue the exact number of clusters is flexible. 
        
        self.bisecting_strategy = bisecting_strategy

        self.max_iter = max_iter
        self.kmeans_kwargs = {'max_iter':max_iter, 'n_clusters':2, 'n_init':n_init, 'random_state':42}

        self.verbose = verbose 
        self.check_non_homogenous = (bisecting_strategy == 'largest_non_homogenous')

    def _is_homogenous(self):
        n_classes_per_cluster = self._get_n_classes_per_cluster()
        return np.all(n_classes_per_cluster == 1)

    def _get_n_non_homogenous(self):
        n_classes_per_cluster = self._get_n_classes_per_cluster()
        return (n_classes_per_cluster > 1).sum()

    def _get_cluster_sizes(self):
        return np.bincount(self.cluster_labels)
    
    def _get_n_classes_per_cluster(self):
        cluster_labels = np.arange(self.curr_cluster_label)
        n_classes_per_cluster = [len(np.unique(self.labels[self.cluster_labels == i])) for i in cluster_labels]
        return np.array(n_classes_per_cluster)
    
    def converged(self, kmeans):
        return kmeans.n_iter_ < self.max_iter
        
    def _get_cluster_to_split(self):

        cluster_sizes = self._get_cluster_sizes()
        assert not np.all(cluster_sizes == 1), 'Clusterer._get_cluster_to_split: There are no non-singleton clusters remaining.'

        if self.bisecting_strategy == 'largest_non_homogenous':
            if self._is_homogenous():
                self.bisecting_strategy = 'largest'
            else:
                n_classes_per_cluster = self._get_n_classes_per_cluster()
                cluster_sizes = np.where(n_classes_per_cluster == 1, 0, cluster_sizes)

        return np.argmax(cluster_sizes)
    
    def transform(self, dataset):
        embeddings = dataset.numpy().astype(np.float16)
        embeddings = self.scaler.transform(embeddings)
        dists = pairwise_distances(embeddings, self.cluster_centers, metric='euclidean')
        return dists 
    
    def predict(self, dataset):
        dists = self.transform(dataset)
        return np.argmin(dists, axis=1)
    
    def fit(self, dataset):

        embeddings = dataset.numpy().astype(np.float16)
        embeddings = self.scaler.fit_transform(embeddings)

        self.labels = dataset.label if hasattr(dataset, 'label') else None 
        self.index = dataset.index 
        self.cluster_labels = np.zeros(len(dataset), dtype=np.int64) # Initialize the cluster labels. 

        iter = 0
        while self.curr_cluster_label < self.n_clusters:
        # for _ in tqdm(desc='Clusterer.fit', total=self.n_clusters - 1):
            cluster_to_split = self._get_cluster_to_split()
            self.split_order.append(cluster_to_split)
            cluster_idxs = np.where(self.cluster_labels == cluster_to_split)[0]

            if self.verbose:
                print(f'Clusterer.fit: Split {iter}, cluster {cluster_to_split} divided using bisection strategy {self.bisecting_strategy}.')
            
            kmeans = KMeans(**self.kmeans_kwargs)
            kmeans.fit(embeddings[cluster_idxs])

            assert self.converged(kmeans), f'Clusterer.fit: The KMeans clusterer did not converge when splitting cluster {cluster_to_split}.'

            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_
            cluster_labels = np.where(cluster_labels == 0, cluster_to_split, self.curr_cluster_label)

            self.cluster_centers[cluster_to_split] = cluster_centers[0]
            self.cluster_centers.append(cluster_centers[1])

            self.cluster_labels[cluster_idxs] = cluster_labels

            pattern = r'(?<!\d)' + str(cluster_to_split) + r'(?!\d)'
            self.tree = re.sub(pattern, f'({cluster_to_split}, {self.curr_cluster_label})', self.tree, count=1) 
            
            self.curr_cluster_label += 1 # Update the current cluster label. 
            iter += 1

        self.cluster_centers = np.array(self.cluster_centers) # Convert to a numpy array. 

        if self.check_non_homogenous:
            n = self._get_n_non_homogenous()
            warnings.warn(f'Clusterer.fit: There are {n} remaining clusters which are not homogenous.')

    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path:str):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj

# class Clusterer():

#     def __init__(self, tolerance=1e-8, n_clusters:int=1000, n_init:int=10, max_iter:int=1000, verbose:bool=False):
        
#         self.n_clusters = n_clusters
#         self.kmeans = BisectingKMeans(verbose=verbose, n_clusters=n_clusters, bisecting_strategy='largest_cluster', tol=tolerance, n_init=n_init, random_state=42, max_iter=max_iter) # Will use Euclidean distance. 
#         self.scaler = StandardScaler() # I think scaling prior to clustering is important. Applying same assumption as with Classifier training. 
#         self.cluster_map = None
#         self.cluster_labels = None
#         self.index = None # Stores the index of the data used to fit the model. 

#     def _check_homogenous(self, dataset):
#         df = pd.DataFrame(index=dataset.index)
#         df['label'] = dataset.label 
#         df['cluster_label'] = self.cluster_labels
#         for cluster_label, cluster_df in df.groupby('cluster_label'):
#             assert cluster_df.label.nunique() == 1, f'Clusterer._check_homogenous: Cluster {cluster_label} is not homogenous.'

#     def transform(self, dataset):
#         embeddings = dataset.numpy().astype(np.float16) # Use half precision to reduce memory. 
#         embeddings = self.scaler.transform(embeddings)
#         dists = self.kmeans.transform(embeddings)
#         return dists 
    
#     def predict(self, dataset):
#         dists = self.transform(dataset)
#         return dists.argmin(axis=1) 
    
#     def fit(self, dataset): # , check_homogenous:bool=False):
#         embeddings = dataset.numpy().astype(np.float16) # Use half precision to reduce memory. 
#         embeddings = self.scaler.fit_transform(embeddings)
#         self.index = dataset.index
#         self.kmeans.fit(embeddings)
#         self.cluster_labels = self.kmeans.labels_ 
#         self.cluster_map = dict(list(zip(self.index, self.cluster_labels)))
#         self.cluster_centers = self.kmeans.cluster_centers_

#         # if check_homogenous and (hasattr(dataset, 'label')):
#         #     self._check_homogenous(dataset)
    






