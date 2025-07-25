import pandas as pd 
import numpy as np 
from scipy.special import comb
from scipy.sparse import lil_array
import itertools
from tqdm import tqdm 
from numpy.linalg import norm
from src.dataset import Dataset
from sklearn.preprocessing import StandardScaler
# from sklearn.utils.validation import check_is_fitted
from scipy.spatial.distance import cdist 


scaler_is_fitted = lambda scaler : hasattr(scaler, 'mean_')


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

        self.lookup_map = None

    def _init_lookup(self):
        # There is new scipy behavior with the csr_array where indexing returns a COO (coordinate) array (not a CSR array).
        # Need to convert explicitly.
        lookup_data = self.matrix[0].tocsr().data 
        lookup_idxs = self.matrix[0].tocsr().indices 
        self.lookup_map = dict(zip(lookup_idxs, lookup_data))

    def _get_index(self, i:int, j:int):
        '''Convert a two-dimensional index to a one-dimensional index.'''
        i, j = min(i, j), max(i, j)
        offset = (i * (2 * self.n - i - 1)) // 2 # I think we need to subtract i so we are back into an index (otherwise gets shifted by one each time)
        # offset = 0 if (i == 0) else sum([self.n - (i_ + 1) - 1 for i_ in range(i)]) # The number of elements before row i, shifted one to the left so that it's an index. 
        return offset + (j - i - 1)
    
    def _get_index_vectorized(self, i:np.ndarray, j:np.ndarray):
        # Indexing breaks down when you try to put elements along the diagonal of the original matrix, as the internal array
        # assumes the diagonal is not being stored
        assert np.all(i != j), 'PackedDistanceMatrix.get_index_vectorized: Should not be trying to access points where i == j.'
        # I timed this function, it is extremely fast. Will not be the cause of any time bottlenecks. 
        i, j = np.minimum(i, j), np.maximum(i, j)
        offset = (i * (2 * self.n - i - 1)) // 2 - i
        idx = offset + (j - 1)
        return idx 
    
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
    
    def _get_vectorized(self, i:np.ndarray, j:np.ndarray):
        # For some reason, everything freaks out when I try to access this with a vector. 
        idxs = self._get_index_vectorized(i, j)
        # values = np.array([self.matrix[0, idx] for idx in idxs])
        # values = self.matrix[0, idxs] # Returns a COO array because of fancy indexing.
        values = np.array([self.lookup_map.get(idx, 0.0) for idx in idxs], dtype=np.float32)
        return values
    
    def toarray(self):
        array = np.zeros((self.n, self.n))
        for i, j in itertools.combinations(range(self.n), 2):
            value = self.get(i, j)
            array[i, j] = value
            array[j, i] = value
        return array
        
    @classmethod
    def from_array(cls, embeddings:np.ndarray, sample_idxs:list=None, batch_size:int=1000):
        n = len(embeddings)
        sample_idxs = np.arange(n) if (sample_idxs is None) else sample_idxs
        matrix = cls(n)

        # When computing the silhouette index, it is going to be necessary to subset the dataset. In this case, I only want to 
        # compute distances between the sampled elements versus all other entries in the dataset. However, I still want to take
        # advantage of symmetry to avoid computing a full (sample_size, n) distance matrix, so I decided to 
        # make the PackedDistanceMatrix sparse. 

        # This computes all pairs of indices for the embeddings for which the distances will be computed. 
        i_idxs, j_idxs = np.meshgrid(np.arange(n), sample_idxs, indexing='ij')
        i_idxs, j_idxs = i_idxs.ravel(), j_idxs.ravel()
        i_idxs, j_idxs = np.minimum(i_idxs, j_idxs), np.maximum(i_idxs, j_idxs)
        idxs = np.unique(np.stack([i_idxs, j_idxs], axis=1), axis=0)
        idxs = idxs[idxs[:, 0] != idxs[:, 1]]

        # Expected number of indices is the sample size choose 2 plus times the number of sample embeddings times the total number of embeddings subtracted by the sample size.
        expected_n_idxs = ((n - len(sample_idxs)) * len(sample_idxs)) + comb(len(sample_idxs), 2)
        assert len(idxs) == expected_n_idxs, f'PackedDistanceMatrix.from_array: Expected {expected_n_idxs}, but saw {len(idxs)}.'

        mem = np.dtype(matrix.dtype).itemsize * len(idxs) / (1024 ** 3)
        print(f'PackedDistanceMatrix.from_array: Adding {len(idxs)} entries to the packed distance matrix, requiring {mem:.3f}GB of memory.', flush=True)

        n_batches = int(np.ceil(len(idxs) / batch_size))
        batched_idxs = np.array_split(idxs, n_batches, axis=0)
        for idxs_ in tqdm(batched_idxs, desc='PackedDistanceMatrix.from_array', file=sys.stdout):
            distances = norm(embeddings[idxs_[:, 0]] - embeddings[idxs_[:, 1]], axis=1)
            matrix._put_vectorized(idxs_[:, 0], idxs_[:, 1], distances)

        matrix.matrix = matrix.matrix.tocsr() # Converting to CSR for much faster read access.
        matrix._init_lookup()
        return matrix
    

class DistanceMatrix():

    def __init__(self, n:int, dtype=np.float32, index:np.ndarray=None):
        
        self.n = n 
        self.matrix = np.zeros((n, n), dtype=dtype)
        self.index = index 

        # 
        # self.scaler = scaler if (scaler is not None) else StandardScaler()


    @classmethod
    def from_dataset(cls, dataset:Dataset, scaler:StandardScaler=None):

        matrix = cls(len(dataset), index=dataset.index)

        assert not dataset.scaled, 'DistanceMatrix.from_dataset: Dataset has already been scaled.'
        embeddings = dataset.numpy()
        assert ((scaler is None) or scaler_is_fitted(scaler)), 'DistanceMatrix.__init__: Input StandardScaler has not been fitted.'
        scaler = StandardScaler() if (scaler is None) else scaler 
        embeddings = scaler.fit_transform(embeddings) if (not scaler_is_fitted(scaler)) else scaler.transform(embeddings)

        matrix.matrix = cdist(embeddings, embeddings, metric='euclidean')
        return matrix
    
    
    def to_df(self):

        df = pd.DataFrame(self.matrix, columns=self.index)
        df['query_id'] = self.index 
        df = df.melt(value_name='distance', var_name='subject_id', id_vars='query_id')
        return df 
    
    def numpy(self):
        return self.matrix 

