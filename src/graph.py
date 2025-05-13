import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix, csr_matrix
import pickle
from tqdm import tqdm
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# TODO: Add checks for setting metadata. 
# TODO: Read about how radius neighbor graphs are constructed. Are all pairwise distances computed?

# Decided to use a radius neighbor graph in leiu of cluster-based analysis.



class NeighborsGraph():

    def __init__(self, radius:float=20, dims:int=100, n_neighbors:int=5):
        self.random_state = 42
        self.radius = radius # Radius is inclusive. 
        self.n_neighbors = n_neighbors
        self.graph = None 
        self.neighbor_idxs = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=dims)
        self.dims = dims
        self.shape = None

        self.metadata = None
        self.id_to_index_map = None # Will store a dictionary mapping IDs to numerical indices. 
        self.index_to_id_map = None # Will store a dictionary mapping numerical indices to sequence IDs. 

    def _preprocess(self, dataset):
        embeddings = dataset.numpy()
        dims = embeddings.shape[-1]
        if dims > self.dims: # Only PCA reduce if the number of dimensions specified at initialization is less than that of the embeddings. 
            embeddings = self.scaler.fit_transform(embeddings)
            embeddings = self.pca.fit_transform(embeddings)
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f'NeighborsGraph._preprocess: Used PCA to reduce dimensions from {dims} to {self.dims}. Total explained variance is {explained_variance:4f}.')
        return embeddings

    def _get_neighbor_idxs(self, id_:str, include_query:bool=False) -> np.ndarray:
        idx = self.id_to_index_map[id_]
        neighbor_idxs = self.neighbor_idxs[idx]
        if include_query:
            return neighbor_idxs
        else: # Don't include the query ID if specified. 
            return idx, neighbor_idxs[neighbor_idxs != idx] 
    
    def get_n_neighbors(self, id_) -> int:
        return len(self._get_neighbor_idxs(id_)[-1])
    
    def get_neighbor_ids(self, id_) -> list:
        _, neighbor_idxs = self._get_neighbor_idxs(id_)
        ids = np.array([self.index_to_id_map[idx] for idx in neighbor_idxs])
        return ids  
    
    def get_neighbor_distances(self, id_:str) -> np.ndarray:
        idx, neighbor_idxs = self._get_neighbor_idxs(id_)
        return self.graph[idx, neighbor_idxs].toarray().ravel()
    
    def get_neighbor_metadata(self, id_:str):
        neighbor_ids = self.get_neighbor_ids(id_)
        metadata_df = self.metadata.loc[neighbor_ids].copy()
        metadata_df[f'distance'] = self.get_neighbor_distances(id_)
        metadata_df = metadata_df.sort_values(f'distance')
        return metadata_df

    def _merge_graphs(self, graphs):
        merged_graph = set()
        for graph in graphs:
            n_nonzero = graph.count_nonzero()
            graph = graph.tocoo() # Convert to COO to access the row, column, and data as separate arrays. 
            merged_graph.update(zip(graph.row, graph.col, graph.data))
            print(f'NeighborsGraph._merge_graphs: Merged a graph with {n_nonzero} nonzero elements. Merged graph now has {len(merged_graph)} entries.')
        rows, cols, data = zip(*merged_graph) # Unpack the tuples into three separate lists. 
        merged_graph = csr_matrix((data, (rows, cols)), shape=self.shape)
        return merged_graph
    
    def _get_edges(self, subset_ids:list=None, weighted:bool=True):
        self.graph.setdiag(0)
        self.graph.eliminate_zeros()
        subset_ids = list(self.id_to_index_map) if (subset_ids is None) else subset_ids

        # Want to get all edges connecting the neighbors to one another as well, so get every node participating in the graph. 
        idxs = np.unique(np.concatenate([self._get_neighbor_idxs(id_, include_query=True) for id_ in subset_ids]).ravel())
        ids = np.array([self.index_to_id_map[idx] for idx in idxs])

        # Accessing sparse arrays doesn't work the same as dense arrays; graph[idxs, idxs] gets the diagonal. 
        graph = self.graph[np.ix_(idxs, idxs)].tocoo(copy=True) # Get as a COO matrix for easy index access. 

        edges = list(zip(ids[graph.row], ids[graph.col]))
        weights = 1 / graph.data if weighted else np.ones(len(edges))
        weights = weights / weights.max() # Normalize the weights so the plot renders correctly. 
        assert (weights == 0).sum() == 0, 'NeighborsGraph._get_edges: Some of the edges have zero weights.'

        return edges, weights

    def _get_graph(self, subset_ids:list=None, colors:dict=dict(), weighted:bool=True) -> nx.Graph:
        graph = nx.Graph()

        edges, weights = self._get_edges(subset_ids=subset_ids, weighted=weighted)

        for (i, j), weight in tqdm(zip(edges, weights), desc='NeighborsGraph.get_graph'):
            graph.add_node(i, color=colors.get(i, 'black'))
            graph.add_node(j, color=colors.get(j, 'black'))
            graph.add_edge(i, j, weight=weight, show=((i in subset_ids) or (j in subset_ids))) # Invert the weight for the spring layout. Weight corresponds to "attractive force."
        return graph
    
    def draw(self, subset_ids:list=None, colors:dict=dict(), ax:plt.Axes=None, weighted:bool=False, **kwargs):

        graph = self._get_graph(subset_ids=subset_ids, colors=colors, weighted=weighted)
        pos = nx.spring_layout(graph, weight='weight')
        colors = [node_data['color'] for _, node_data in graph.nodes(data=True)]
        alphas = [int(edge_data['show']) for _, _, edge_data in graph.edges(data=True)]
        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=colors, node_size=kwargs.get('node_size', 20))
        nx.draw_networkx_edges(graph, pos=pos, ax=ax, alpha=alphas)
        ax.set_axis_off()
        return graph


    def fit(self, dataset):

        self.shape = (len(dataset), len(dataset))
        embeddings = self._preprocess(dataset) # Make sure the embeddings are scaled. 
        self.metadata = dataset.metadata()
        self.id_to_index_map = {id_:i for i, id_ in enumerate(dataset.index)}
        self.index_to_id_map = {i:id_ for i, id_ in enumerate(dataset.index)}

        print(f'NeighborsGraph.fit: Fitting the NearestNeighbors object with radius {self.radius} and {self.n_neighbors} neighbors.', flush=True)
        nearest_neighbors = NearestNeighbors(metric='euclidean', radius=self.radius, n_neighbors=self.n_neighbors)
        nearest_neighbors.fit(embeddings)
        
        # Density of embedded points is very uneven, so using only radius neighbors results in some points having a ton of neighbors, and others having very few. 
        # Instead, opted to merge the results of two graph types, so that the space around the dense points is sufficiently well-characterized, but there 
        # are still points of comparison for the points which don't have any nearby neighbors. 
        graphs = list()
        print(f'NeighborsGraph.fit: Building the radius neighbors graph with radius {self.radius}.', flush=True)
        graphs += [nearest_neighbors.radius_neighbors_graph(X=embeddings, radius=self.radius, mode='distance', sort_results=True)] # Output is a CSR sparse matrix. 
        print(f'NeighborsGraph.fit: Building the k-neighbors graph with {self.n_neighbors} neighbors.', flush=True)
        graphs += [nearest_neighbors.kneighbors_graph(X=embeddings, n_neighbors=self.n_neighbors, mode='distance')] # Output is a CSR sparse matrix. 
        self.graph = self._merge_graphs(graphs)

        neighbor_idxs = list()
        neighbor_idxs.append(nearest_neighbors.radius_neighbors(embeddings, radius=self.radius, return_distance=False))
        neighbor_idxs.append(nearest_neighbors.kneighbors(embeddings, n_neighbors=self.n_neighbors, return_distance=False))
        self.neighbor_idxs = [np.unique(np.concatenate(idxs)) for idxs in zip(*neighbor_idxs)]

    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path:str):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj