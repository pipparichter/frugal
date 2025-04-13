import pandas as pd 
import numpy as np 
from sklearn.cluster import BisectingKMeans, KMeans # , OPTICS
from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
import warnings 
import torch 
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances, euclidean_distances
from tqdm import tqdm
import pickle
import re 
import math 
import sys 
import itertools
from sklearn.metrics import silhouette_score

# TODO: Is there any data leakage by fitting a StandardScaler before clustering? I don't think any more so
#   than caused by clustering the training and testing dataset together. 
# TODO: Why does KMeans need to use Euclidean distance? Other distance metrics also have a concept of closeness. 
# https://www.biorxiv.org/content/10.1101/2024.11.13.623527v1.full
# TODO: Can and should probably make tree parsing recursive.

# TODO: Add check to make sure Dataset and Clusterer indices match. 


# Can choose to calculate intra-cluster distance in one of three ways: mean distance between all pairs, the 
# mean distance of all points from the mean, and the maximum distance between any two points in the cluster. 

# Can choose to calculate inter-cluster distance in one of three ways: the closest two points between clusters, 
# the farthest two points between clusters, or the distance between centroids. 

# The Dunn index is then computed as the minimum inter-cluster distance between any two clusters divided by the maximum
# intra-cluster distance of any cluster. 
            

class PackedDistanceMatrix():
    def __init__(self, n:int, dtype=np.float16):
        # Because I am ignoring the diagonal, basically have a situation with (n - 1) rows, and each column 
        # decreases by one element. So row 0 has (n - 1) elements, row 1 has (n - 2) elements, etc. 
        self.n = n
        self.dtype = dtype
        self.size = math.comb(n, 2)

        mem = np.dtype(self.dtype).itemsize * self.size / (1024 ** 3)
        print(f'PackedDistanceMatrix.__init__: Allocating {mem:.3f}GB of memory.', flush=True)
        self.matrix = np.zeros(self.size, dtype=dtype)

    def _get_index(self, i:int, j:int):
        '''Convert a two-dimensional index to a one-dimensional index.'''
        # Number of elements in row i is (n - (i + 1)). Because j > i, j is always greater than 0. 
        return int(i * (self.n - (i + 1)) + (j - 1))

    def get(self, i:int, j:int):
        if i == j:
            return 0
        return self.matrix[self._get_index(min(i, j), max(i, j))]
    
    def put(self, i:int, j:int, value:np.float16):
        if i == j:
            return 
        self.matrix[self._get_index(min(i, j), max(i, j))] = value.astype(np.float16)

    @classmethod
    def from_embeddings(cls, embeddings:np.ndarray):
        n = len(embeddings)
        matrix = cls(n)
        pbar = tqdm(list(itertools.combinations(np.arange(n), 2)), desc='PackedDistanceMatrix.from_embeddings', file=sys.stdout)
        for i, j in pbar:
            matrix.put(i, j, euclidean(embeddings[i], embeddings[j]))
            # matrix.put(i, j, pairwise_distances(np.expand_dims(embeddings[i], axis=0), np.expand_dims(embeddings[j], axis=0), metric='euclidean'))
        pbar.close()
        return matrix
    

def check_packed_distance_matrix(embeddings):
    D_ = pairwise_distances(embeddings, metric='euclidean')
    D = PackedDistanceMatrix.from_embeddings(embeddings)
    n = len(embeddings)
    for i in range(n):
        for j in range(n):
            assert D.get(i, j) == D_[i, j], f'check_packed_distance_matrix: Distances do not agree. Expected {D_[i, j]}, got {D.get(i, j)}.'
    

class Clusterer():

    def __init__(self, verbose:bool=True, n_clusters:int=1000, n_init:int=10, max_iter:int=1000, bisecting_strategy:str='largest_non_homogenous'):
        
        self.n_clusters = n_clusters 
        self.scaler = StandardScaler() # I think scaling prior to clustering is important. Applying same assumption as with Classifier training. 
        self.index = None # Stores the index of the data used to fit the model. 
        
        self.curr_cluster_id = 1
        self.tree = '0'
        self.split_order = []

        self.cluster_ids = None
        self.labels = None
        self.cluster_centers = [None] # Store as a list, becaue the exact number of clusters is flexible. 
        self.cluster_idxs = None # Dictionary mapping cluster IDs to the dataset indices. 
        
        self.bisecting_strategy = bisecting_strategy

        self.max_iter = max_iter
        self.kmeans_kwargs = {'max_iter':max_iter, 'n_clusters':2, 'n_init':n_init, 'random_state':42}

        self.verbose = verbose 
        self.check_non_homogenous = (bisecting_strategy == 'largest_non_homogenous')

    def _is_homogenous(self):
        n_labels_per_cluster = self._get_n_labels_per_cluster()
        return np.all(n_labels_per_cluster == 1)

    def _get_n_non_homogenous(self):
        n_labels_per_cluster = self._get_n_labels_per_cluster()
        return (n_labels_per_cluster > 1).sum()

    def _get_cluster_sizes(self):
        return np.bincount(self.cluster_ids)
    
    def _get_n_labels_per_cluster(self):
        cluster_ids = np.arange(self.curr_cluster_id)
        n_labels_per_cluster = [len(np.unique(self.labels[self.cluster_ids == i])) for i in cluster_ids]
        return np.array(n_labels_per_cluster)
    
    def get_cluster_id_to_label_map(self):
        assert (self.labels is not None), 'Clusterer.get_cluster_id_to_label_map: Clusterer object has no stored labels.'
        labels = [self.labels[self.cluster_ids == i][0] for i in np.arange(self.n_clusters)]
        return dict(zip(np.arange(self.n_clusters), labels))
    
    def converged(self, kmeans):
        return kmeans.n_iter_ < self.max_iter
    
    def fitted(self):
        '''Return whether or not the Clusterer has been fitted.'''
        return (self.cluster_ids is not None)
        
    def _get_cluster_to_split(self):

        cluster_sizes = self._get_cluster_sizes()
        assert not np.all(cluster_sizes == 1), 'Clusterer._get_cluster_to_split: There are no non-singleton clusters remaining.'

        if self.bisecting_strategy == 'largest_non_homogenous':
            if self._is_homogenous():
                self.bisecting_strategy = 'largest'
            else:
                n_labels_per_cluster = self._get_n_labels_per_cluster()
                cluster_sizes = np.where(n_labels_per_cluster == 1, 0, cluster_sizes)

        return np.argmax(cluster_sizes)
    
    def subset(self, idxs:np.ndarray):

        cluster_ids = self.cluster_ids[idxs].copy()
        labels = self.labels[idxs].copy()
        index = self.index[idxs].copy()
        cluster_idxs = {i:np.where(cluster_ids == i)[0] for i in np.unique(cluster_ids)}
        n_clusters = len(np.unique(cluster_ids))

        clusterer = Clusterer(n_clusters=n_clusters, bisecting_strategy=self.bisecting_strategy)
        clusterer.scaler = self.scaler 
        clusterer.cluster_idxs = cluster_idxs
        clusterer.cluster_ids = cluster_ids
        clusterer.labels = labels
        clusterer.index = index

        return clusterer

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
        self.cluster_ids = np.zeros(len(dataset), dtype=np.int64) # Initialize the cluster labels. 

        iter = 0
        while self.curr_cluster_id < self.n_clusters:
            cluster_to_split = self._get_cluster_to_split()
            self.split_order.append(cluster_to_split)
            cluster_idxs = np.where(self.cluster_ids == cluster_to_split)[0]

            if self.verbose:
                print(f'Clusterer.fit: Split {iter}, cluster {cluster_to_split} divided using bisection strategy {self.bisecting_strategy}.', flush=True)
            
            kmeans = KMeans(**self.kmeans_kwargs)
            kmeans.fit(embeddings[cluster_idxs])

            assert self.converged(kmeans), f'Clusterer.fit: The KMeans clusterer did not converge when splitting cluster {cluster_to_split}.'

            cluster_ids = np.where(kmeans.labels_ == 0, cluster_to_split, self.curr_cluster_id)
            cluster_centers = kmeans.cluster_centers_

            self.cluster_centers[cluster_to_split] = cluster_centers[0] # Update the center of the split cluster. 
            self.cluster_centers.append(cluster_centers[1]) # Add the cluster center of the new cluster. 
            self.cluster_ids[cluster_idxs] = cluster_ids

            pattern = r'(?<!\d)' + str(cluster_to_split) + r'(?!\d)'
            self.tree = re.sub(pattern, f'({cluster_to_split}, {self.curr_cluster_id})', self.tree, count=1) 
            
            self.curr_cluster_id += 1 # Update the current cluster label. 
            iter += 1

        self.cluster_centers = np.array(self.cluster_centers) # Convert to a numpy array. 
        self.cluster_idxs = {i:np.where(self.cluster_ids == i)[0] for i in range(self.n_clusters)}

        if self.check_non_homogenous:
            n = self._get_n_non_homogenous()
            if (n > 0):
                warnings.warn(f'Clusterer.fit: There are {n} remaining clusters which are not homogenous.')

    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path:str):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    
    def _check_dataset(self, dataset):
        assert len(dataset.index) == len(self.index), 'Clusterer._check_dataset: Dataset and cluster indices do not match.'
        assert np.all(dataset.index == self.index), 'Clusterer._check_dataset: Dataset and cluster indices do not match.'
        assert np.all(dataset.cluster_id == self.cluster_ids), 'Clusterer._check_dataset: Datased and cluster indices do not match.'
    
    def _get_intra_cluster_distance(self, i:int, method:str='center', embeddings:np.ndarray=None):
        cluster_embeddings = embeddings[self.cluster_idxs[i]]
        if method == 'center':
            cluster_center = np.expand_dims(self.cluster_centers[i], axis=0)
            distances = pairwise_distances(cluster_center, cluster_embeddings, metric='euclidean')
            return distances.mean()
        distances = pairwise_distances(cluster_embeddings, metric='euclidean')
        if method == 'pairwise':
            return distances[np.triu_indices(distances, k=1)].mean(axis=None)
        if method == 'furthest':
            return distances.max(axis=None)
        
    def _get_inter_cluster_distance(self, i:int, j:int, embeddings:np.ndarray=None, method:str='center'):
        if method == 'center':
            cluster_center_i, cluster_center_j = np.expand_dims(self.cluster_centers[i]), np.expand_dims(self.cluster_centers[j])
            distances = pairwise_distances(cluster_center_i, cluster_center_j, metric='euclidean')
            return distances.mean()
        cluster_i_embeddings, cluster_j_embeddings = embeddings[self.cluster_idxs[i]], embeddings[self.cluster_idxs[j]]
        distances = pairwise_distances(cluster_i_embeddings, cluster_j_embeddings, metric='euclidean')
        if method == 'closest':
            return distances.min(axis=None)
        if method == 'furthest':
            return distances.max(axis=None)
        
    def get_silhouette_index(self, dataset):
        self._check_dataset(dataset)
        embeddings = self.scaler.transform(dataset.numpy()).astype(np.float16)
        cluster_metadata_df = pd.DataFrame(index=np.arange(self.n_clusters), columns=['silhouette_index', 'silhouette_index_weight']) # There is a good chance that not every cluster will be represented. 
        cluster_sizes = np.bincount(self.cluster_ids)
        print(silhouette_score(embeddings, self.cluster_ids))

        check_packed_distance_matrix(embeddings)
        D = PackedDistanceMatrix.from_embeddings(embeddings)

        def a(x, i:int):
            d = np.array([D.get(x, y) for y in self.cluster_idxs[i]])
            return d[d > 0].mean(axis=None) # Remove the one x_i to x_i distance, which will be zero. 

        def b(x, i:int):
            '''For a datapoint x in cluster i, compute the mean distance between x and all elements in cluster j. 
            Then, return the minimum of these mean distances over all clusters i != j.'''
            d = lambda j : np.array([D.get(x, y) for y in self.cluster_idxs[j]]).mean(axis=None)
            return min([d(j) for j in np.unique(self.cluster_ids) if (j != i)])
        
        def s(x, i:int):
            if cluster_sizes[i] == 1:
                return 0
            else:
                a_x, b_x = a(x, i), b(x, i)
                return (b_x - a_x) / max(a_x, b_x)
        
        silhouette_index = {i:list() for i in np.unique(self.cluster_ids)}
        for x in tqdm(range(len(embeddings)), desc='Clusterer.get_silhouette_index', file=sys.stdout):
            i = self.cluster_ids[x]
            silhouette_index[i].append(s(x, i))
        
        for i in silhouette_index.keys():
            cluster_metadata_df.loc[i, 'silhouette_index'] = np.mean(silhouette_index[i])
            cluster_metadata_df.loc[i, 'silhouette_index_weight'] = len(silhouette_index[i])
        silhouette_index = [value for values in silhouette_index.values() for value in values] # Unravel the silhouette values. 
        silhouette_index = np.array(silhouette_index).mean(axis=None)

        return silhouette_index, cluster_metadata_df

    def get_dunn_index(self, dataset, inter_method:str='center', intra_method:str='center'):
        '''https://en.wikipedia.org/wiki/Dunn_index'''
        self._check_dataset(dataset)
        embeddings = self.scaler.transform(dataset.numpy()).astype(np.float16)
        cluster_metadata_df = pd.DataFrame(index=np.arange(self.n_clusters), columns=[f'intra_cluster_distance_{intra_method}', f'min_inter_cluster_distance_{inter_method}'])

        for i in tqdm(range(self.n_clusters), desc='get_dunn_index'):
            inter_cluster_distances = [self._get_inter_cluster_distance(i, j, embeddings=embeddings, method=inter_method) for j in range(self.n_clusters) if (i != j)]
            cluster_metadata_df.loc[i, f'intra_cluster_distance_{intra_method}'] = self._get_intra_cluster_distance(i, embeddings=embeddings, method=intra_method)
            cluster_metadata_df.loc[i, f'min_inter_cluster_distance_{inter_method}'] = min(inter_cluster_distances)

        dunn_index = cluster_metadata_df[f'intra_cluster_distance_{intra_method}'].max()
        dunn_index /= cluster_metadata_df[f'min_inter_cluster_distance_{inter_method}'].min()

        return dunn_index, cluster_metadata_df
    
    def get_davies_bouldin_index(self, dataset):
        self._check_dataset(dataset)
        embeddings = self.scaler.transform(dataset.numpy()).astype(np.float16)
        cluster_metadata_df = pd.DataFrame(index=np.arange(self.n_clusters), columns=[f'intra_cluster_distance_center', 'davies_bouldin_index'])
 
        D = pairwise_distances(self.cluster_centers, metric='euclidean') # Might need to do this one at a time if memory is a problem. 
        sigma = np.array([self._get_intra_cluster_distance(i, embeddings=embeddings) for i in range(self.n_clusters)])
        for i in range(self.n_clusters):
            cluster_metadata_df['davies_bouldin_index'] = max([(sigma[i] + sigma[j]) / D[i, j] for j in range(self.n_clusters) if (i != j)])
        cluster_metadata_df['intra_cluster_distance_center'] = sigma 
        davies_bouldin_index = cluster_metadata_df['davies_bouldin_index'].sum() / self.n_clusters

        return davies_bouldin_index, cluster_metadata_df


class ClusterTreeNode():

    def __init__(self, cluster_ids:list, index:np.ndarray=None, labels:np.ndarray=None):

        self.index = index
        self.labels = labels
        self.cluster_ids = cluster_ids

        self.children = []

    def is_homogenous(self):
        return np.all(self.labels == self.labels[0])
    
    def contains(self, cluster_id:int):
        return (cluster_id in self.cluster_ids)

    def is_terminal(self):
        return len(self.children) == 0

    def __len__(self):
        return len(self.cluster_ids)


class ClusterTree():
    split_pattern = r'\(([\d]{1,}), ([\d]{1,})\)'
    def __init__(self, clusterer):

        self.tree = clusterer.tree 
        self.cluster_ids = clusterer.cluster_ids
        self.index = clusterer.index
        self.labels = clusterer.labels
        self.n_splits = clusterer.n_clusters - 1
        self.root_node = self._parse_tree(self.tree)
    
    def _get_node(self, cluster_ids:list):
        mask = np.isin(self.cluster_ids, cluster_ids)
        index = self.index[mask].copy()
        labels = self.labels[mask].copy()
        return ClusterTreeNode(cluster_ids=cluster_ids, index=index, labels=labels)
    
    def _merge_nodes(self, left_node, right_node):
        cluster_ids = left_node.cluster_ids + right_node.cluster_ids
        parent_node = self._get_node(cluster_ids)
        parent_node.children = [left_node, right_node]
        return parent_node
    
    def _parse_tree(self, tree:str):

        nodes = {cluster_id:self._get_node([cluster_id]) for cluster_id in np.unique(self.cluster_ids)}

        pbar = tqdm(total=self.n_splits, desc='ClusterTree._parse_tree')
        while len(tree) > 1:
            splits = re.findall(ClusterTree.split_pattern, tree)
            splits = [(int(left_node_label), int(right_node_label)) for left_node_label, right_node_label in splits]

            for left_node_label, right_node_label in splits:

                assert left_node_label < right_node_label, 'ClusterTree._parse_tree: Expected left node label to be less than right node label.'

                right_node, left_node = nodes[left_node_label], nodes[right_node_label]
                parent_node = self._merge_nodes(left_node, right_node)
                
                nodes[left_node_label] = parent_node
                nodes[right_node_label] = None 
                tree = tree.replace(f'({left_node_label}, {right_node_label})', str(left_node_label))
                pbar.update(1)

        # assert len(nodes) == 1, 'Clusterer._parse_tree: There should only be one node in in the nodes dictionary.'
        pbar.close()
        return nodes[0]
    
    def get_first_homogenous_node(self, cluster_id:int=None):
        '''Retrieve the first node containing the specified cluster which is homogenous, as well as the depth at which
        the node is found. The number of splits required to get a homogenous cluster containing the cluster label
        should be a proxy for the cluster "difficulty."'''

        def _get_first_homogenous_node(curr_node:ClusterTreeNode, n_splits:int):
            if curr_node.is_homogenous():
                return curr_node, n_splits 
            elif curr_node.is_terminal():
                print(f'ClusterTree.get_first_homogenous_node: No homogenous node containing {cluster_id} was found in the tree.')
                return None, n_splits
            else:
                left_node, right_node = curr_node.children
                next_node = left_node if (left_node.contains(cluster_id)) else right_node
                return _get_first_homogenous_node(next_node, n_splits + 1)
            
        return _get_first_homogenous_node(self.root_node, 0)
        






