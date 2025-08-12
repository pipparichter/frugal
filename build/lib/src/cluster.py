import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans # , OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestNeighbors
import warnings 
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import pickle
import re 
import hnswlib
import sys 
from src.distance import PackedDistanceMatrix

# TODO: Is there any data leakage by fitting a StandardScaler before clustering? I don't think any more so
#   than caused by clustering the training and testing dataset together. 
# TODO: Why does KMeans need to use Euclidean distance? Other distance metrics also have a concept of closeness. 
# https://www.biorxiv.org/content/10.1101/2024.11.13.623527v1.full
# TODO: Can and should probably make tree parsing recursive.
# TODO: Clustering with 50,000 clusters results in a clustering with a pretty low silhouette index, I think in part because of
#   how many clusters there are. This is not necessarily concerning (because the goal is even sampling of the space, not strong clusters), 
#   but I am curious if setting fewer n_clusters results in a better clustering.
# TODO: Make sure sub-sample for silhouette score is stratified by cluster. 
        
    

class Clusterer():

    intra_dist_methods = ['center', 'pairwise', 'furthest']
    inter_dist_methods = ['center', 'furthest', 'closest']

    def __init__(self, dims:int=None, n_clusters:int=10000, n_init:int=10, max_iter:int=1000, bisecting_strategy:str='largest_non_homogenous'):
        
        self.dims = dims
        self.n_clusters = n_clusters 
        self.random_state = 42 # Setting a consistent random state and keeping all other parameters the same results in reproducible clustering output. 

        self.pca = PCA(n_components=dims, random_state=self.random_state) if (dims is not None) else None
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
        self.n_init = n_init
        self.kmeans_kwargs = {'max_iter':max_iter, 'n_clusters':2, 'n_init':n_init, 'random_state':self.random_state}

        self.check_non_homogenous = (bisecting_strategy == 'largest_non_homogenous')

    def _preprocess(self, dataset, fit:bool=False):
        embeddings = dataset.numpy()
        if fit:
            dims = embeddings.shape[-1]
            embeddings = self.scaler.fit_transform(embeddings)
            if self.pca is not None:
                embeddings = self.pca.fit_transform(embeddings)
                explained_variance = self.pca.explained_variance_ratio_.sum()
                print(f'Clusterer._preprocess: Used PCA to reduce dimensions from {dims} to {self.dims}. Total explained variance is {explained_variance:4f}.')
        else:
            embeddings = self.scaler.transform(embeddings)
            if self.pca is not None:
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
    
    # Functions for computing cluster metrics. I wrote my own implementations for these metrics to allow more flexibility, e.g. 
    # for storing per-cluster metrics and working with the PackedDistanceMatrix for more efficient memory usage (I hope). 

    def _get_sample_idxs(self, sample_size:int=None):
        '''Get indices for a sub-sample. If stratified is set to true, ensures the sample contains an even spread of the clusters.'''

        sample_idxs = np.random.choice(np.arange(len(self.index)), size=sample_size, replace=False)
        print(f'Clusterer._get_sample_idxs: {len(np.unique(self.cluster_ids[sample_idxs]))} out of {self.n_clusters} clusters represented in the sample.')
        return sample_idxs
    
    def _check_dataset(self, dataset):
        assert len(dataset.index) == len(self.index), 'Clusterer._check_dataset: Dataset and cluster indices do not match.'
        assert np.all(dataset.index == self.index), 'Clusterer._check_dataset: Dataset and cluster indices do not match.'
        assert np.all(dataset.cluster_id == self.cluster_ids), 'Clusterer._check_dataset: Datased and cluster indices do not match.'





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
    
    # def _get_nearest_cluster_ids(self, M:int=15, ef_construction:int=100, k:int=5):
    #     print('Clusterer._get_nearest_cluster_ids: Initializing HNSW index for nearby cluster searches.')
    #     hnsw = hnswlib.Index(space='l2', dim=self.cluster_centers.shape[-1])
    #     hnsw.init_index(max_elements=self.n_clusters, M=M, ef_construction=ef_construction)
    #     hnsw.set_ef(50)
    #     hnsw.add_items(self.cluster_centers)
    #     nearest_cluster_ids, _ = hnsw.knn_query(self.cluster_centers, k=k)
    #     # Make sure to exclude the first result, which is apparently just the cluster itself.
    #     return nearest_cluster_ids[:, 1:] # Returns an (N, k) array with the labels of the nearest clusters. 

    
    # def _get_intra_cluster_distance(self, i:int, method:str='center', embeddings:np.ndarray=None):
    #     cluster_embeddings = embeddings[self.cluster_idxs[i]]
    #     if len(cluster_embeddings) == 1:
    #         return 0
    #     if method == 'center':
    #         cluster_center = np.expand_dims(self.cluster_centers[i], axis=0)
    #         distances = pairwise_distances(cluster_center, cluster_embeddings, metric='euclidean')
    #         return distances.mean()
    #     distances = pairwise_distances(cluster_embeddings, metric='euclidean')
    #     if method == 'pairwise':
    #         return distances[np.triu_indices(len(distances), k=1)].mean(axis=None)
    #     if method == 'furthest':
    #         return distances.max(axis=None)
        
    # def _get_inter_cluster_distance(self, i:int, j:int, embeddings:np.ndarray=None, method:str='center'):
    #     if method == 'center':
    #         cluster_center_i, cluster_center_j = np.expand_dims(self.cluster_centers[i], axis=0), np.expand_dims(self.cluster_centers[j], axis=0)
    #         distances = pairwise_distances(cluster_center_i, cluster_center_j, metric='euclidean')
    #         return distances.mean(axis=None)
    #     cluster_i_embeddings, cluster_j_embeddings = embeddings[self.cluster_idxs[i]], embeddings[self.cluster_idxs[j]]
    #     distances = pairwise_distances(cluster_i_embeddings, cluster_j_embeddings, metric='euclidean')
    #     if method == 'closest':
    #         return distances.min(axis=None)
    #     if method == 'furthest':
    #         return distances.max(axis=None)
        
    # def _init_cluster_metadata(self, cols:list=[]):
    #     return pd.DataFrame(index=pd.Index(np.arange(self.n_clusters), name='cluster_id'), columns=cols)
        
    # def get_min_inter_cluster_distance(self, dataset, method:str=None):
    #     self._check_dataset(dataset)
    #     embeddings = self._preprocess(dataset, fit=False)
    #     cluster_metadata_df = self._init_cluster_metadata([f'min_inter_cluster_distance_{method}'])

    #     # Get the k nearest clusters to avoid computing inter-cluster distances between every cluster.  
    #     nearest_cluster_ids = self._get_nearest_cluster_ids(k=20)
    #     for i in tqdm(np.arange(self.n_clusters), desc='Clusterer.get_min_inter_cluster_distance'):
    #         inter_cluster_distances = [self._get_inter_cluster_distance(i, j, embeddings=embeddings, method=method) for j in nearest_cluster_ids[i]]
    #         cluster_metadata_df.loc[i, f'min_inter_cluster_distance_{method}'] = min(inter_cluster_distances)

    #     min_inter_cluster_distance = cluster_metadata_df[f'min_inter_cluster_distance_{method}'].mean()
    #     return min_inter_cluster_distance, cluster_metadata_df

    # def get_intra_cluster_distance(self, dataset, method:str=None):
    #     self._check_dataset(dataset)
    #     embeddings = self._preprocess(dataset, fit=False)
    #     cluster_metadata_df = self._init_cluster_metadata([f'intra_cluster_distance_{method}'])

    #     for i in tqdm(np.arange(self.n_clusters), desc='Clusterer.get_intra_cluster_distance'):
    #         cluster_metadata_df.loc[i, f'intra_cluster_distance_{method}'] = self._get_intra_cluster_distance(i, method=method, embeddings=embeddings)

    #     intra_cluster_distance = cluster_metadata_df[f'min_inter_cluster_distance_{method}'].mean() # Use the mean of the intra-cluster distances as a summary statistic.
    #     return intra_cluster_distance, cluster_metadata_df
        
    # def get_silhouette_index(self, dataset, sample_size:int=None):
    #     '''A silhouette index for a particular point x in cluster i is essentially the "closeness" of point x to the nearest cluster i != j minus the
    #     "closeness" of x to its own cluster i, normalized according to the maximum of the two closeness metrics. A negative silhoutte index therefore
    #     implies that x is closer to another cluster than its own cluster, while a score close to zero indicates that a point is equidistant between
    #     two clusters. Silhouette indices range from -1 to 1. https://en.wikipedia.org/wiki/Silhouette_(clustering)'''
    #     self._check_dataset(dataset)
    #     embeddings = self._preprocess(dataset, fit=False)
    #     cluster_metadata_df = self._init_cluster_metadata(['silhouette_index', 'silhouette_index_weight'])

    #     cluster_sizes = np.bincount(self.cluster_ids) 
    #     sample_idxs = np.arange(len(self.index)) if (sample_size is None) else self._get_sample_idxs(sample_size=sample_size)
    #     nearest_cluster_ids = self._get_nearest_cluster_ids(k=20)

    #     D = PackedDistanceMatrix.from_array(embeddings, sample_idxs=sample_idxs)

    #     def a(x, i:int):
    #         '''For a datapoint in cluster i, compute the mean distance from all elements in cluster i.'''
    #         cluster_idxs_i = self.cluster_idxs[i]
    #         cluster_idxs_i = cluster_idxs_i[cluster_idxs_i != x]
    #         d = D._get_vectorized(np.repeat(x, len(cluster_idxs_i)), cluster_idxs_i)
    #         # Ran into an issue here where I am taking the mean of an empty slice.
    #         assert (d > 0).sum() > 0, f'Clusterer.get_silhouette_index: All intra-cluster distances are zero in cluster {i} of size {cluster_sizes[i]}.'
    #         return d[d > 0].mean(axis=None) # Remove the one x_i to x_i distance, which will be zero. 

    #     def b(x, i:int):
    #         '''For a datapoint x in cluster i, compute the mean distance from all elements in cluster j, and return the minimum.'''
    #         d = lambda j : D._get_vectorized(np.repeat(x, cluster_sizes[j]), self.cluster_idxs[j]).mean(axis=None)
    #         return min([d(j) for j in nearest_cluster_ids[i] if (j != i)])
        
    #     def s(x, i:int):
    #         if cluster_sizes[i] == 1:
    #             return 0 # Silhouette index is not well-defined for singleton clusters. 
    #         a_x = a(x, i)
    #         b_x = b(x, i)
    #         return (b_x - a_x) / max(a_x, b_x)
        
    #     silhouette_index = dict() # Store silhouette score computations by cluster. 
    #     # for i_, x in enumerate(sample_idxs): 
    #     for x in tqdm(sample_idxs, desc='Clusterer.get_silhouette_index', file=sys.stdout):
    #         i = self.cluster_ids[x]
    #         if i not in silhouette_index:
    #             silhouette_index[i] = []
    #         s_x = s(x, i)
    #         assert s_x != np.nan, 'Clusterer.get_silhouette_index: Computed a NaN silhouette index.'
    #         # print(f'Clusterer.get_silhouette_index: Computed silhouette index of {s_x:.4f} for element {i_} of {len(sample_idxs)}.', flush=True)
    #         silhouette_index[i].append(s_x)
        
    #     for i in silhouette_index.keys():
    #         cluster_metadata_df.loc[i, 'silhouette_index_mean'] = np.mean(silhouette_index[i])
    #         cluster_metadata_df.loc[i, 'silhouette_index_weight'] = len(silhouette_index[i])
    #         # cluster_metadata_df.loc[i, 'silhouette_index_min'] = np.min(silhouette_index[i])
    #         # cluster_metadata_df.loc[i, 'silhouette_index_max'] = np.max(silhouette_index[i])
    #         # cluster_metadata_df.loc[i, 'silhouette_index_n_negative'] = (np.array(silhouette_index[i]) < 0).sum()
    #     silhouette_index = [value for values in silhouette_index.values() for value in values] # Unravel the silhouette values. 
    #     silhouette_index = np.array(silhouette_index).mean(axis=None)

    #     return silhouette_index, cluster_metadata_df

    # def get_dunn_index(self, dataset, inter_dist_method:str='center', intra_dist_method:str='center'):
    #     '''https://en.wikipedia.org/wiki/Dunn_index'''
    #     self._check_dataset(dataset)
    #     embeddings = self._preprocess(dataset, fit=False)
    #     cluster_metadata_df = self._init_cluster_metadata([f'intra_cluster_distance_{intra_dist_method}', f'min_inter_cluster_distance_{inter_dist_method}'])

    #     for i in tqdm(range(self.n_clusters), desc='get_dunn_index'):
    #         inter_cluster_distances = [self._get_inter_cluster_distance(i, j, embeddings=embeddings, method=inter_dist_method) for j in range(self.n_clusters) if (i != j)]
    #         cluster_metadata_df.loc[i, f'intra_cluster_distance_{intra_dist_method}'] = self._get_intra_cluster_distance(i, embeddings=embeddings, method=intra_dist_method)
    #         cluster_metadata_df.loc[i, f'min_inter_cluster_distance_{inter_dist_method}'] = min(inter_cluster_distances)

    #     dunn_index = cluster_metadata_df[f'intra_cluster_distance_{intra_dist_method}'].max()
    #     dunn_index /= cluster_metadata_df[f'min_inter_cluster_distance_{inter_dist_method}'].min()

    #     return dunn_index, cluster_metadata_df
    
    # def get_davies_bouldin_index(self, dataset):
    #     '''To compute the Davies-Bouldin index for a specific cluster i, first compute the intra-cluster distance (measured as the mean distance of each
    #     point to the cluster center) for cluster i. Then, compute the intra-cluster distance for every other cluster j != i, and divide the sum by the
    #     distance between the i and j cluster centers. Take the maximum of these values; the maximum value is for the cluster pair i, j for which the separation
    #     is the worst.'''
    #     self._check_dataset(dataset)
    #     embeddings = self._preprocess(dataset, fit=False)
    #     cluster_metadata_df = self._init_cluster_metadata([f'intra_cluster_distance_center', 'davies_bouldin_index'])
    #     nearest_cluster_ids = self._get_nearest_cluster_ids(k=10)
    #     k = nearest_cluster_ids.shape[-1] # Number of nearest clusters (will be 19)

    #     # Pre-compute pairwise distances between cluster centers, as well as intra-cluster distances. 
    #     D = PackedDistanceMatrix.from_array(self.cluster_centers) # Might need to do this one at a time if memory is a problem. 
    #     print(f'Clusterer.get_davies_bouldin_index: Computed pairwise distances between cluster centers.', flush=True)
    #     sigma = np.array([self._get_intra_cluster_distance(i, embeddings=embeddings, method='center') for i in range(self.n_clusters)])
    #     print(f'Clusterer.get_davies_bouldin_index: Computed intra-cluster distances using method "center."', flush=True)

    #     for i in tqdm(range(self.n_clusters), desc='Clusterer.get_davies_bouldin_index'):
    #         sigma_i = np.repeat(sigma[i], k)
    #         sigma_j = sigma[nearest_cluster_ids[i]]
    #         distances = D._get_vectorized(np.repeat(i, k).astype(int), nearest_cluster_ids[i].astype(int))
    #         cluster_metadata_df.loc[i, 'davies_bouldin_index'] = ((sigma_i + sigma_j) / distances).max(axis=None)
    #     cluster_metadata_df['intra_cluster_distance_center'] = sigma 
    #     davies_bouldin_index = cluster_metadata_df['davies_bouldin_index'].sum() / self.n_clusters

    #     return davies_bouldin_index, cluster_metadata_df



