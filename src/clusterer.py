import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans # , OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestNeighbors
import warnings 
from sklearn.metrics import pairwise_distances
from numpy.linalg import norm
from tqdm import tqdm
import pickle
from scipy.sparse import lil_array
import re 
import hnswlib
from scipy.special import comb
import sys 
import itertools
import time 

# TODO: Is there any data leakage by fitting a StandardScaler before clustering? I don't think any more so
#   than caused by clustering the training and testing dataset together. 
# TODO: Why does KMeans need to use Euclidean distance? Other distance metrics also have a concept of closeness. 
# https://www.biorxiv.org/content/10.1101/2024.11.13.623527v1.full
# TODO: Can and should probably make tree parsing recursive.
# TODO: Clustering with 50,000 clusters results in a clustering with a pretty low silhouette index, I think in part because of
#   how many clusters there are. This is not necessarily concerning (because the goal is even sampling of the space, not strong clusters), 
#   but I am curious if setting fewer n_clusters results in a better clustering.
# TODO: Make sure sub-sample for silhouette score is stratified by cluster. 
            

class PackedDistanceMatrix():
    def __init__(self, n:int, dtype=np.float32):
        # Because I am ignoring the diagonal, basically have a situation with (n - 1) rows, and each column 
        # decreases by one element. So row 0 has (n - 1) elements, row 1 has (n - 2) elements, etc. 
        self.n = n
        self.dtype = dtype
        self.size = int(comb(n, 2, exact=False, repetition=False))
        mem = np.dtype(self.dtype).itemsize * self.size / (1024 ** 3)
        print(f'PackedDistanceMatrix.__init__: Packed distance matrix will require at most {self.size} elements, requiring {mem:.3f}GB of memory.', flush=True)
        self.matrix = lil_array((1, self.size), dtype=dtype) # Storing as a sparse array to efficiently handle computing distance matrices for sub-samples.


    def _get_index(self, i:int, j:int):
        '''Convert a two-dimensional index to a one-dimensional index.'''
        i, j = min(i, j), max(i, j)
        offset = (i * (2 * self.n - i - 1)) // 2 - i # I think we need to subtract i so we are back into an index (otherwise gets shifted by one each time)
        # offset = 0 if (i == 0) else sum([self.n - (i_ + 1) - 1 for i_ in range(i)]) # The number of elements before row i, shifted one to the left so that it's an index. 
        return offset + (j - 1)
    
    def _get_index_vectorized(self, i:np.ndarray, j:np.ndarray):
        i, j = np.minimum(i, j), np.maximum(i, j)
        offset = (i * (2 * self.n - i - 1)) // 2 - i
        return offset + (j - 1)

    def get(self, i:int, j:int):
        if i == j:
            return 0
        return self.matrix[0, self._get_index(i, j)]
    
    def put(self, i:int, j:int, value):
        if i == j:
            return 
        self.matrix[0, self._get_index(i, j)] = value
    
    def _put_vectorized(self, i:np.ndarray, j:np.ndarray, values:np.ndarray):
        self.matrix[0, self._get_index_vectorized(i, j)] = values
    
    def _get_vectorized(self, i: np.ndarray, j: np.ndarray):
        # For some reason, everything freaks out when I try to access this with a vector. 
        idxs = self._get_index_vectorized(i, j)
        t1 = time.perf_counter()
        values = np.array([self.matrix[0, idx] for idx in idxs])
        t2 = time.perf_counter()
        print(f'PackedDistanceMatrix._get_vectorized: Retrieved {len(idxs)} elements from the matrix in {t2 - t1:.4f} seconds.', flush=True)
        return values
        
    @classmethod
    def from_array(cls, embeddings:np.ndarray, sample_idxs:list=None, batch_size:int=1000):
        n = len(embeddings)
        matrix = cls(n)

        # When computing the silhouette index, it is going to be necessary to subset the dataset. In this case, I only want to 
        # compute distances between the sampled elements versus all other entries in the dataset. However, I still want to take
        # advantage of symmetry to avoid computing a full (sample_size, n) distance matrix, so I decided to 
        # make the PackedDistanceMatrix sparse. 
        if sample_idxs is not None: # Only compute distances relative to an index in the sample subset.
            i_idxs, j_idxs = np.meshgrid(np.arange(n), sample_idxs, indexing='ij')
            i_idxs, j_idxs = i_idxs.ravel(), j_idxs.ravel()
            i_idxs, j_idxs = np.minimum(i_idxs, j_idxs), np.maximum(i_idxs, j_idxs)
            idxs = np.unique(np.stack([i_idxs, j_idxs], axis=1), axis=0)
        else:
            idxs = list(itertools.combinations(np.arange(n), 2))
            idxs = np.array(idxs)

        mem = np.dtype(matrix.dtype).itemsize * len(idxs) / (1024 ** 3)
        print(f'PackedDistanceMatrix.__init__: Adding {len(idxs)} entries to the packed distance matrix, requiring {mem:.3f}GB of memory.', flush=True)

        n_batches = int(np.ceil(len(idxs) / batch_size))
        batched_idxs = np.array_split(idxs, n_batches, axis=0)
        for idxs_ in tqdm(batched_idxs, desc='PackedDistanceMatrix.from_array', file=sys.stdout):
            distances = norm(embeddings[idxs_[:, 0]] - embeddings[idxs_[:, 1]], axis=1)
            matrix._put_vectorized(idxs_[:, 0], idxs_[:, 1], distances)

        matrix.matrix = matrix.matrix.tocsr() # Converting to CSR for much faster read access.
        return matrix
    

# def check_packed_distance_matrix(embeddings):
#     D_ = pairwise_distances(embeddings, metric='euclidean')
#     D = PackedDistanceMatrix.from_array(embeddings)
#     n = len(embeddings)
#     for i in range(n):
#         for j in range(n):
#             assert np.isclose(D.get(i, j), D_[i, j], atol=1e-5), f'check_packed_distance_matrix: Distances do not agree at ({i}, {j}). Expected {D_[i, j]}, got {D.get(i, j)}.'
#             # print(f'check_packed_distance_matrix: Distances agree at ({i}, {j}).')
    

class Clusterer():

    def __init__(self, dims:int=20, n_clusters:int=10000, n_init:int=10, max_iter:int=1000, bisecting_strategy:str='largest_non_homogenous'):
        
        self.dims = dims
        self.n_clusters = n_clusters 
        self.random_state = 42 # Setting a consistent random state and keeping all other parameters the same results in reproducible clustering output. 

        self.pca = PCA(n_components=dims, random_state=self.random_state)
        self.scaler = StandardScaler() # I think scaling prior to clustering is important. Applying same assumption as with Classifier training. 

        self.curr_cluster_id = 1
        self.tree = '0'

        self.index = None # Stores the index of the data used to fit the model. 
        self.labels = None 
        self.cluster_ids = None
        self.labels = None
        self.cluster_centers = [None] # Store as a list, becaue the exact number of clusters is flexible. 
        self.cluster_idxs = None # Dictionary mapping cluster IDs to the dataset indices. 
        
        self.bisecting_strategy = bisecting_strategy

        self.max_iter = max_iter
        self.n_init
        self.kmeans_kwargs = {'max_iter':max_iter, 'n_clusters':2, 'n_init':n_init, 'random_state':self.random_state}

        self.check_non_homogenous = (bisecting_strategy == 'largest_non_homogenous')

    def _preprocess(self, dataset, fit:bool=False):
        embeddings = dataset.numpy()
        if fit:
            dims = embeddings.shape[-1]
            embeddings = self.scaler.fit_transform(embeddings)
            embeddings = self.pca.fit_transform(embeddings)
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f'Clusterer._preprocess: Used PCA to reduce dimensions from {dims} to {self.dims}. Total explained variance is {explained_variance:4f}.')
        else:
            embeddings = self.scaler.transform(embeddings)
            embeddings = self.pca.transform(embeddings)
        return embeddings

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
        embeddings = self._preprocess(dataset, fit=False)
        dists = pairwise_distances(embeddings, self.cluster_centers, metric='euclidean')
        return dists 
    
    def predict(self, dataset):
        dists = self.transform(dataset)
        return np.argmin(dists, axis=1)
    
    def fit(self, dataset):

        embeddings = self._preprocess(dataset, fit=True)

        self.labels = dataset.label.copy() if hasattr(dataset, 'label') else None 
        self.index = dataset.index.copy()
        self.cluster_ids = np.zeros(len(dataset), dtype=np.int64) # Initialize the cluster labels. 

        iter = 0
        while self.curr_cluster_id < self.n_clusters:
            cluster_to_split = self._get_cluster_to_split()
            cluster_idxs = np.where(self.cluster_ids == cluster_to_split)[0]

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
    
    # Functions for computing cluster metrics. 

    def _get_sample_idxs(self, sample_size:int=None):
        '''Get indices for a sub-sample. If stratified is set to true, ensures the sample contains an even spread of the clusters.'''

        sample_idxs = np.random.choice(np.arange(len(self.index)), size=sample_size, replace=False)
        print(f'Clusterer._get_sample_idxs: {len(np.unique(self.cluster_ids[sample_idxs]))} out of {self.n_clusters} clusters represented in the sample.')
        return sample_idxs
    
    def _check_dataset(self, dataset):
        assert len(dataset.index) == len(self.index), 'Clusterer._check_dataset: Dataset and cluster indices do not match.'
        assert np.all(dataset.index == self.index), 'Clusterer._check_dataset: Dataset and cluster indices do not match.'
        assert np.all(dataset.cluster_id == self.cluster_ids), 'Clusterer._check_dataset: Datased and cluster indices do not match.'

    def _get_nearest_cluster_ids(self, M:int=15, ef_construction:int=100, k:int=5):
        print('Clusterer._get_nearest_cluster_ids: Initializing HNSW index for nearby cluster searches.')
        hnsw = hnswlib.Index(space='l2', dim=self.cluster_centers.shape[-1])
        hnsw.init_index(max_elements=self.n_clusters, M=M, ef_construction=ef_construction)
        hnsw.set_ef(50)
        hnsw.add_items(self.cluster_centers)
        nearest_cluster_ids, _ = hnsw.knn_query(self.cluster_centers, k=k)
        return nearest_cluster_ids # Returns an (N, k) array with the labels of the nearest clusters. 

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
        
    def get_silhouette_index(self, dataset, sample_size:int=None):
        '''A silhouette index for a particular point x in cluster i is essentially the "closeness" of point x to the nearest cluster i != j minus the
        "closeness" of x to its own cluster i, normalized according to the maximum of the two closeness metrics. A negative silhoutte index therefore
        implies that x is closer to another cluster than its own cluster, while a score close to zero indicates that a point is equidistant between
        two clusters. Silhouette indices range from -1 to 1. https://en.wikipedia.org/wiki/Silhouette_(clustering)'''
        
        self._check_dataset(dataset)
        
        embeddings = self.scaler.transform(dataset.numpy()).astype(np.float32)
        cluster_metadata_df = pd.DataFrame(index=np.arange(self.n_clusters), columns=['silhouette_index', 'silhouette_index_weight']) # There is a good chance that not every cluster will be represented. 
        # check_packed_distance_matrix(embeddings)

        cluster_sizes = np.bincount(self.cluster_ids) 
        sample_idxs = np.arange(len(self.index)) if (sample_size is None) else self._get_sample_idxs(sample_size=sample_size)
        nearest_cluster_ids = self._get_nearest_cluster_ids(k=20)

        D = PackedDistanceMatrix.from_array(embeddings, sample_idxs=sample_idxs)

        def a(x, i:int):
            '''For a datapoint in cluster i, compute the mean distance from all elements in cluster i.'''
            d = D._get_vectorized(np.repeat(x, cluster_sizes[i]), self.cluster_idxs[i])
            assert len(d) == cluster_sizes[i], 'Clusterer.get_silhouette_index: The number of intra-cluster distances should be equal to the cluster size.'
            return d[d > 0].mean(axis=None) # Remove the one x_i to x_i distance, which will be zero. 

        def b(x, i:int):
            '''For a datapoint x in cluster i, compute the mean distance from all elements in cluster j, and return the minimum.'''
            d = lambda j : D._get_vectorized(np.repeat(x, cluster_sizes[j]), self.cluster_idxs[j]).mean(axis=None)
            return min([d(j) for j in nearest_cluster_ids[i] if (j != i)])
        
        def s(x, i:int):
            if cluster_sizes[i] == 1:
                return 0 # Silhouette index is not well-defined for singleton clusters. 
            
            t1 = time.perf_counter()
            a_x = a(x, i)
            t2 = time.perf_counter()
            print(f'Clusterer.get_silhouette_index: Time for computing a(x, i) on x={x} is {t2 - t1:.4f} seconds.', flush=True)
            
            t1 = time.perf_counter()
            b_x = b(x, i)
            t2 = time.perf_counter()
            print(f'Clusterer.get_silhouette_index: Time for computing b(x, i) on x={x} is {t2 - t1:.4f} seconds.', flush=True)

            return (b_x - a_x) / max(a_x, b_x)
        
        # print('Clusterer.get_silhouette_index: Beginning silhouette index calculation.')
        silhouette_index = dict() # Store silhouette score computations by cluster. 
        for i_, x in enumerate(sample_idxs): 
        # for x in tqdm(sample_idxs, desc='Clusterer.get_silhouette_index', file=sys.stdout):
            i = self.cluster_ids[x]
            if i not in silhouette_index:
                silhouette_index[i] = []
            s_x = s(x, i)
            print(f'Clusterer.get_silhouette_index: Computed silhouette index of {s_x:.4f} for element {i_} of {len(sample_idxs)}.', flush=True)
            silhouette_index[i].append(s_x)
        
        for i in silhouette_index.keys():
            cluster_metadata_df.loc[i, 'silhouette_index_mean'] = np.mean(silhouette_index[i])
            cluster_metadata_df.loc[i, 'silhouette_index_weight'] = len(silhouette_index[i])
            # cluster_metadata_df.loc[i, 'silhouette_index_min'] = np.min(silhouette_index[i])
            # cluster_metadata_df.loc[i, 'silhouette_index_max'] = np.max(silhouette_index[i])
            # cluster_metadata_df.loc[i, 'silhouette_index_n_negative'] = (np.array(silhouette_index[i]) < 0).sum()
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
 
        D = PackedDistanceMatrix.from_array(self.cluster_centers) # Might need to do this one at a time if memory is a problem. 
        
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
        


    # def subset(self, idxs:np.ndarray):

    #     cluster_ids = self.cluster_ids[idxs].copy() # Get the subset new cluster ID array. 
    #     n_clusters = len(np.unique(cluster_ids))
    #     print(f'Clusterer.subset: {n_clusters} of the {self.n_clusters} original clusters are represented in the subset.')

    #     clusterer = Clusterer(n_clusters=n_clusters, bisecting_strategy=self.bisecting_strategy, max_iter=self.max_iter, n_int=self.n_init)
    #     clusterer.scaler = self.scaler # Copy over the fitted scaler. 
    #     clusterer.cluster_idxs = {i:np.where(cluster_ids == i)[0] for i in np.unique(cluster_ids)} # Re-compute the cluster-to-index map with the new subset.
    #     clusterer.cluster_ids = cluster_ids
    #     clusterer.labels = self.labels[idxs].copy()
    #     clusterer.index = self.index[idxs].copy()
    #     return clusterer



        # if stratified:
        #     if sample_size < self.n_clusters:
        #         print(f'Clusterer._get_sample_idxs: Sample size is too small. Using the minimum sample size of {self.n_clusters}.')
        #         sample_size = self.n_clusters # Can't sample fewer than the number of clusters. 
        #     splits = ClusterStratifiedShuffleSplit(dataset, n_splits=1, train_size=(sample_size / len(dataset)))
        #     sample_idxs, _ = list(splits)[0]
        # else:


