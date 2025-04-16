import unittest 
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from src.dataset import Dataset
from src.clusterer import Clusterer, PackedDistanceMatrix
import pandas as pd
import numpy as np 


class TestClusterer(unittest.TestCase):
    dataset_df = pd.read_csv('/home/prichter/Documents/frugal/tests/data/dataset_subset.csv', index_col=0)
    dataset = Dataset.from_hdf('/home/prichter/Documents/frugal/tests/data/dataset_subset.h5', attrs=['cluster_id', 'label'])
    clusterer = Clusterer.load('/home/prichter/Documents/frugal/tests/data/dataset_subset_cluster.pkl')
    embeddings = clusterer._preprocess(dataset, fit=False)
    cluster_ids = clusterer.cluster_ids

    def test_silhouette_index(self):
        silhouette_index, _ = TestClusterer.clusterer.get_silhouette_index(TestClusterer.dataset, sample_size=None)
        self.assertAlmostEqual(silhouette_index, silhouette_score(TestClusterer.embeddings, TestClusterer.cluster_ids), delta=0.01)

    def test_davies_bouldin_pairwise_center_distances(self):

        D = PackedDistanceMatrix.from_array(TestClusterer.clusterer.cluster_centers).toarray()
        D_ = pairwise_distances(TestClusterer.clusterer.cluster_centers, metric='euclidean')
        self.assertTrue(np.all(np.isclose(D, D_)))

    def test_davies_bouldin_index(self):
        davies_bouldin_index, _ = TestClusterer.clusterer.get_davies_bouldin_index(TestClusterer.dataset)
        self.assertAlmostEqual(davies_bouldin_index, davies_bouldin_score(TestClusterer.embeddings, TestClusterer.cluster_ids), delta=0.01)


class TestPackedDistanceMatrix(unittest.TestCase):

    def test_distances_match_sklearn_pairwise_distances(self):
        pass 

    
    # I was running into problems where the memory usage seemed to explode when I tried to access the underlying
    # CSR matrix with a vector of indices. I want to get a sense of what's going on. 
    # def test_expected_memory_usage_with_get_vectorized(self):
    #     pass 


# def check_packed_distance_matrix(embeddings):
#     D_ = pairwise_distances(embeddings, metric='euclidean')
#     D = PackedDistanceMatrix.from_array(embeddings)
#     n = len(embeddings)
#     for i in range(n):
#         for j in range(n):
#             assert np.isclose(D.get(i, j), D_[i, j], atol=1e-5), f'check_packed_distance_matrix: Distances do not agree at ({i}, {j}). Expected {D_[i, j]}, got {D.get(i, j)}.'
#             # print(f'check_packed_distance_matrix: Distances agree at ({i}, {j}).')
    
if __name__ == '__main__':

    unittest.main()