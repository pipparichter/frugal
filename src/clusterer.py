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

# TODO: Is there any data leakage by fitting a StandardScaler before clustering? I don't think any more so
#   than caused by clustering the training and testing dataset together. 
# TODO: Why does KMeans need to use Euclidean distance? Other distance metrics also have a concept of closeness. 
# https://www.biorxiv.org/content/10.1101/2024.11.13.623527v1.full
# TODO: Can and should probably make tree parsing recursive.

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
        pbar.close()
        return matrix


def get_scaled_embeddings(dataset, clusterer):
    '''Extract the embeddings from the Dataset and apply the StandardScaler stored in the Clusterer object.'''
    embeddings = dataset.numpy() 
    embeddings = clusterer.scaler.transform(embeddings).astype(np.float16) # Half-precision to reduce memory. 
    embeddings_df = pd.DataFrame(embeddings, index=pd.Index(dataset.index, name='id'))
    return embeddings_df


def get_silhouette_index(dataset, clusterer, sample_size:int=None):
    
    # Will be far more efficient to pre-compute all distances, but will take a lot of memory.   
    embeddings_df = get_scaled_embeddings(dataset, clusterer)
    cluster_ids = clusterer.cluster_ids 
    n = len(np.unique(cluster_ids))

    if (sample_size is not None):
        sample_idxs = np.random.choice(len(dataset), sample_size, replace=False)
        embeddings_df = embeddings_df.iloc[sample_idxs].copy()
        cluster_ids = cluster_ids[sample_idxs].copy()
        n = len(np.unique(cluster_ids))
    
    cluster_idxs = {i:np.where(cluster_ids == i)[0] for i in cluster_ids}
    cluster_sizes = np.bincount(cluster_ids)

    D = PackedDistanceMatrix.from_embeddings(embeddings_df.values)

    def a(x, i:int):
        d = np.array([D.get(x, y) for y in cluster_idxs[i]])
        # d = pairwise_distances(x, embeddings_df.iloc[cluster_idxs[i]])
        return d[d > 0].mean(axis=None) # Remove the one x_i to x_i distance, which will be zero. 

    def b(x, i:int):
        '''For a datapoint x in cluster i, compute the mean distance between x and all elements in cluster j. 
        Then, return the minimum of these mean distances over all clusters i != j.'''
        d = np.inf 
        for j in range(n):
            if i == j:
                continue 
            d_ = np.array([D.get(x, y) for y in cluster_idxs[j]]).mean(axis=None)
            # d_ = pairwise_distances(x, embeddings_df.iloc[cluster_idxs[j]]).mean(axis=None)
            d = min(d_, d)
        return d
    
    def s(x, i:int):
        if cluster_sizes[i] == 1:
            return 0
        else:
            a_x, b_x = a(x, i), b(x, i)
            return (b_x - a_x) / max(a_x, b_x)
    
    s_tilde = list()
    for x in tqdm(range(len(embeddings_df)), desc='get_silhouette_index', file=sys.stdout):
        # x = np.expand_dims(embeddings_df.iloc[idx].values, axis=0)
        i = cluster_ids[x]
        s_tilde.append(s(x, i))

    return np.mean(s_tilde)


def get_dunn_index(dataset):
    pass 

def get_davies_bouldin_index(dataset, clusterer):

    embeddings_df = get_scaled_embeddings(dataset, clusterer)
    n = clusterer.n_clusters 
    cluster_idxs = {i:np.where(clusterer.cluster_ids == i)[0] for i in range(n_clusters)}

    def get_sigma(c, i:int):
        '''Compute the average distance between all elements in cluster i to c_i.'''
        distances = pairwise_distances(np.expand_dims(c[i]), embeddings_df.iloc[cluster_idxs[i]], metric='euclidean')
        return distances.mean(axis=None)
    
    c = clusterer.cluster_centers
    d = pairwise_distances(c, c, metric='euclidean') # Might need to do this one at a time if memory is a problem. 
    sigma = np.array([get_sigma(c, i) for i in range(n)])

    sum_ = 0
    for i in range(n):
        sum_ += max([(sigma[i] + sigma[j]) / d[i, j] for j in range(n_clusters) if (i != j)])

    return sum_ / n



def get_cluster_metadata(dataset, clusterer):

    assert hasattr(dataset, 'cluster_id'), 'get_cluster_distances: There are no cluster IDs associated with the dataset.'
    assert hasattr(dataset, 'label'), 'get_cluster_distances: There are no labels associated with the dataset.'

    cluster_metadata_df = list()

    cluster_df = dataset.metadata(attrs=['cluster_id', 'label'])
    embeddings_df = get_scaled_embeddings(dataset, clusterer)

    pbar = tqdm(list(cluster_df.groupby('cluster_id')), desc='get_cluster_metadata')
    for cluster_id, df in pbar:
        cluster_center = np.expand_dims(clusterer.cluster_centers[cluster_id], axis=0).astype(np.float16)
        cluster_embeddings = embeddings_df.loc[df.index].values

        row = dict()
        row['cluster_id'] = cluster_id 
        row['cluster_size'] = len(df)
        row['cluster_label'] = df['label'].values[0]

        if len(df) > 1:
            idxs = np.triu_indices(len(df), k=1)
            intra_cluster_distances = pairwise_distances(cluster_embeddings, metric='euclidean')[idxs].astype(np.float16)
            row['cluster_radius'] = pairwise_distances(cluster_center, cluster_embeddings, metric='euclidean').astype(np.float16).max(axis=None)
            row['intra_cluster_max_distance'] = intra_cluster_distances.max(axis=None)
            row['intra_cluster_min_distance'] = intra_cluster_distances.min(axis=None)
            row['intra_cluster_mean_distance'] = intra_cluster_distances.mean(axis=None)
        
        idxs = np.array([i for i in range(clusterer.n_clusters) if (i != cluster_id)])
        inter_cluster_distances = pairwise_distances(cluster_center, np.array(clusterer.cluster_centers), metric='euclidean')[:, idxs].astype(np.float16)
        row['inter_cluster_max_distance'] = inter_cluster_distances.max(axis=None)
        row['inter_cluster_min_distance'] = inter_cluster_distances.min(axis=None)
        row['inter_cluster_mean_distance'] = inter_cluster_distances.mean(axis=None)
        cluster_metadata_df.append(row)
    pbar.close()

    cluster_metadata_df = pd.DataFrame(cluster_metadata_df).set_index('cluster_id')
    cluster_metadata_df = cluster_metadata_df.fillna(0.0)
    return cluster_metadata_df



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
        # for _ in tqdm(desc='Clusterer.fit', total=self.n_clusters - 1):
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
    

# Want some kind of metric for cluster "difficulty" based on the bisection tree. 
# Something like "How many splits before the leaf containing the child is homogenous?"
# More splits would mean it's more mixed in with opposite-labeled things. 
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
        


#     def __init__(self, tolerance=1e-8, n_clusters:int=1000, n_init:int=10, max_iter:int=1000, verbose:bool=False):
        
#         self.n_clusters = n_clusters
#         self.kmeans = BisectingKMeans(verbose=verbose, n_clusters=n_clusters, bisecting_strategy='largest_cluster', tol=tolerance, n_init=n_init, random_state=42, max_iter=max_iter) # Will use Euclidean distance. 
#         self.scaler = StandardScaler() # I think scaling prior to clustering is important. Applying same assumption as with Classifier training. 
#         self.cluster_map = None
#         self.cluster_ids = None
#         self.index = None # Stores the index of the data used to fit the model. 

#     def _check_homogenous(self, dataset):
#         df = pd.DataFrame(index=dataset.index)
#         df['label'] = dataset.label 
#         df['cluster_id'] = self.cluster_ids
#         for cluster_id, cluster_df in df.groupby('cluster_id'):
#             assert cluster_df.label.nunique() == 1, f'Clusterer._check_homogenous: Cluster {cluster_id} is not homogenous.'

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
#         self.cluster_ids = self.kmeans.labels_ 
#         self.cluster_map = dict(list(zip(self.index, self.cluster_ids)))
#         self.cluster_centers = self.kmeans.cluster_centers_

#         # if check_homogenous and (hasattr(dataset, 'label')):
#         #     self._check_homogenous(dataset)
    






