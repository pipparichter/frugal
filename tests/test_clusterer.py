import unittest 
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from src.dataset import Dataset
from src.clusterer import Clusterer
import pandas as pd

class TestClusterer(unittest.TestCase):
    dataset_df = pd.read_csv('/home/prichter/Documents/frugal/tests/data/dataset_subset.csv', index_col=0)
    dataset = Dataset.from_hdf('/home/prichter/Documents/frugal/tests/data/dataset_subset.h5', attrs=['cluster_id', 'label'])
    clusterer = Clusterer.load('/home/prichter/Documents/frugal/tests/data/dataset_subset_cluster.pkl')

    def test_silhouette_index(self):
        silhouette_index, 


class TestPackedDistanceMatrix(unittest.TestCase):

    def test_distances_match_sklearn_pairwise_distances(self):
        pass 

    
    # I was running into problems where the memory usage seemed to explode when I tried to access the underlying
    # CSR matrix with a vector of indices. I want to get a sense of what's going on. 
    def test_expected_memory_usage_with_get_vectorized(self):
        pass 


# def check_packed_distance_matrix(embeddings):
#     D_ = pairwise_distances(embeddings, metric='euclidean')
#     D = PackedDistanceMatrix.from_array(embeddings)
#     n = len(embeddings)
#     for i in range(n):
#         for j in range(n):
#             assert np.isclose(D.get(i, j), D_[i, j], atol=1e-5), f'check_packed_distance_matrix: Distances do not agree at ({i}, {j}). Expected {D_[i, j]}, got {D.get(i, j)}.'
#             # print(f'check_packed_distance_matrix: Distances agree at ({i}, {j}).')
    