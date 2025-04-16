import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import pickle
from tqdm import tqdm
import pandas as pd

# TODO: Add checks for setting metadata. 
# TODO: Read about how radius neighbor graphs are constructed. Are all pairwise distances computed?

# Decided to use a radius neighbor graph in leiu of cluster-based analysis.



class RadiusNeighborsGraph():

    def __init__(self, radius:float=15, dims:int=100):
        self.random_state = 42
        self.radius = radius # Radius is inclusive. 
        self.graph = None 
        self.neighbor_idxs = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=dims)
        self.dims = dims

        self.metadata = None
        self.id_to_index_map = None # Will store a dictionary mapping IDs to numerical indices. 
        self.index_to_id_map = None # Will store a dictionary mapping numerical indices to sequence IDs. 

    def _preprocess(self, dataset):
        embeddings = dataset.numpy()
        dims = embeddings.shape[-1]
        embeddings = self.scaler.fit_transform(embeddings)
        embeddings = self.pca.fit_transform(embeddings)
        explained_variance = self.pca.explained_variance_ratio_.sum()
        print(f'RadiusNeighborsGraph._preprocess: Used PCA to reduce dimensions from {dims} to {self.dims}. Total explained variance is {explained_variance:4f}.')

    def _get_neighbor_idxs(self, id_:str) -> np.ndarray:
        idx = self.id_to_index_map[id_]
        neighbor_idxs = self.neighbor_idxs[idx]
        return idx, neighbor_idxs 
    
    def _get_neighbor_distances(self, id_:str) -> np.ndarray:
        idx, neighbor_idxs = self._get_neighbor_idxs(id_)
        return np.array([self.graph[idx, neighbor_idx] for neighbor_idx in neighbor_idxs]) 
    
    def _get_neighbor_idxs_max_distance(self, id_:str, max_distance:float=None) -> tuple:
        idx, neighbor_idxs = self._get_neighbor_idxs(id_)
        neighbor_distances = self._get_neighbor_distances(id_)
        return idx, neighbor_idxs[neighbor_distances < max_distance]

    def _get_neighbor_idxs_max_k(self, id_:str, k:int=None) -> tuple:
        # Because I set sort_results=True, the neighbors should be in order of closest to furthest. 
        idx, neighbor_idxs = self._get_neighbor_idxs(id_)
        n = len(neighbor_idxs)
        return idx, neighbor_idxs[:min(k, n)] # Just in case k is larger than the number of neighbors. 
    
    def get_n_neighbors(self, id_) -> int:
        return len(self._get_neighbor_idxs(id_)[-1])
    
    def get_neighbor_ids(self, id_, max_k:int=None, max_distance:float=None) -> list:

        if (max_k is not None):
            _, neighbor_idxs = self._get_neighbor_idxs_max_k(id_, max_k=max_k)
        elif (max_distance is not None):
            _, neighbor_idxs = self._get_neighbor_idxs_max_distance(id_, max_distance=max_distance)
        else:
            _, neighbor_idxs = self._get_neighbor_idxs(id_)
        return [self.index_to_id_map[idx] for idx in neighbor_idxs]
    
    def get_neighbor_metadata(self, id_:str, field:str=None, max_k:int=None, max_distance:float=None):
        neighbor_ids = self.get_neighbor_ids(id_, max_k=max_k, max_distance=max_distance)
        metadata_df = self.metadata.loc[neighbor_ids].copy()
        if field is not None:
            return metadata_df[field].values
        return metadata_df

    def fit(self, dataset):

        embeddings = self._preprocess(dataset) # Make sure the embeddings are scaled. 
        self.metadata = dataset.metadata()
        self.id_to_index_map = {id_:i for i, id_ in enumerate(dataset.index)}
        self.index_to_id_map = {i:id_ for i, id_ in enumerate(dataset.index)}

        print(f'RadiusNeighborsGraph.fit: Fitting the NearestNeighbors object with radius {self.radius}.')
        nearest_neighbors = NearestNeighbors(metric='euclidean', radius=self.radius)
        nearest_neighbors.fit(embeddings)
        
        print(f'RadiusNeighborsGraph.fit: Building the radius neighbors graph.')
        self.graph = nearest_neighbors.radius_neighbors_graph(X=embeddings, radius=self.radius, mode='distance', sort_results=True) # Output is a CSR sparse matrix. 
        _, self.neighbor_idxs = nearest_neighbors.radius_neighbors(embeddings, return_distance=True, radius=self.radius, sort_results=True)

    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path:str):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj