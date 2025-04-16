import unittest 
from src.split import ClusterStratifiedShuffleSplit
from src.dataset import Dataset
import pandas as pd 
import numpy as np 
from parameterized import parameterized 

# dataset_df = pd.read_csv('../data/datasets/dataset.csv', index_col=0)
# dataset_df = dataset_df.merge(pd.read_csv('../data/datasets/dataset_cluster.csv', index_col=0), right_index=True, left_index=True)
# dataset_subset_df = dataset_df.sample(1000, random_state=42)
# dataset_subset_df.to_csv('../tests/data/dataset_subset.csv')

class TestClusterStratifiedShuffleSplit(unittest.TestCase):

    test_size, train_size = 0.2, 0.8
    dataset_df = pd.read_csv('/home/prichter/Documents/frugal/tests/data/dataset_subset.csv', index_col=0)
    dataset = Dataset.from_hdf('/home/prichter/Documents/frugal/tests/data/dataset_subset.h5', attrs=['cluster_id', 'label'])
    splitter = ClusterStratifiedShuffleSplit(dataset, n_splits=5, train_size=train_size, test_size=test_size)
    splits = list(splitter)

    def __init__(self, *args, **kwargs):
        super(TestClusterStratifiedShuffleSplit, self).__init__(*args, **kwargs)

        # print(f'TestClusterStratifiedShuffleSplit.__init__: {len(TestClusterStratifiedShuffleSplit.dataset_df)} total elements in the Dataset.')
        self.non_singleton_cluster_ids, self.non_singleton_ids = TestClusterStratifiedShuffleSplit.get_non_singleton_ids(TestClusterStratifiedShuffleSplit.dataset_df)
        self.singleton_cluster_ids, self.singleton_ids = TestClusterStratifiedShuffleSplit.get_singleton_ids(TestClusterStratifiedShuffleSplit.dataset_df)

    @staticmethod
    def get_singleton_ids(dataset_df:pd.DataFrame):
        cluster_sizes = dataset_df.groupby('cluster_id').apply(len, include_groups=False)
        singleton_cluster_ids = cluster_sizes.index[cluster_sizes == 1].values 
        singleton_ids = dataset_df[dataset_df.cluster_id.isin(singleton_cluster_ids)].index.values 
        # print(f'TestClusterStratifiedShuffleSplit.get_singleton_ids: Found {len(singleton_ids)} sequences in singleton clusters.')
        return singleton_cluster_ids, singleton_ids

    @staticmethod
    def get_non_singleton_ids(dataset_df:pd.DataFrame):
        cluster_sizes = dataset_df.groupby('cluster_id').apply(len, include_groups=False)
        non_singleton_cluster_ids = cluster_sizes.index[cluster_sizes > 1].values 
        non_singleton_ids = dataset_df[dataset_df.cluster_id.isin(non_singleton_cluster_ids)].index.values 
        # print(f'TestClusterStratifiedShuffleSplit.get_non_singleton_ids: Found {len(non_singleton_ids)} sequences in {len(non_singleton_cluster_ids)} non-singleton clusters.')
        return non_singleton_cluster_ids, non_singleton_ids

    def test_splitter_stores_correct_singleton_ids(self):
        singleton_idxs = TestClusterStratifiedShuffleSplit.splitter.singleton_idxs # Get the singleton indices stored by the splitter. 
        singleton_ids = TestClusterStratifiedShuffleSplit.dataset.index[singleton_idxs]
        self.assertTrue(TestClusterStratifiedShuffleSplit.splitter.n_singleton == len(self.singleton_ids)) 
        self.assertTrue(set(singleton_ids) == set(self.singleton_ids))

    def test_splitter_stores_correct_non_singleton_ids(self):
        non_singleton_idxs = TestClusterStratifiedShuffleSplit.splitter.non_singleton_idxs # Get the singleton indices stored by the splitter. 
        non_singleton_ids = TestClusterStratifiedShuffleSplit.dataset.index[non_singleton_idxs]
        self.assertTrue(TestClusterStratifiedShuffleSplit.splitter.n_non_singleton == len(self.non_singleton_ids)) 
        self.assertTrue(set(non_singleton_ids) == set(self.non_singleton_ids))

    @parameterized.expand(splits)
    def test_no_singletons_in_test_dataset(self, train_dataset, test_dataset):
        self.assertTrue(not np.any(np.isin(test_dataset.index, self.singleton_ids)))
        self.assertTrue(np.all(np.isin(test_dataset.index, self.non_singleton_ids)))

    @parameterized.expand(splits)
    def test_all_singletons_in_train_dataset(self, train_dataset, test_dataset):
        self.assertTrue(np.all(np.isin(self.singleton_ids, train_dataset.index)))

    @parameterized.expand(splits)
    def test_adjusted_split_ratios_are_correct(self, train_dataset, test_dataset):
        train_size, test_size = TestClusterStratifiedShuffleSplit.splitter.train_size, TestClusterStratifiedShuffleSplit.splitter.test_size
        self.assertTrue(train_size == len(train_dataset))
        self.assertTrue(test_size == len(test_dataset))

    @parameterized.expand(splits)
    def test_no_overlap_between_train_test_datasets(self, train_dataset, test_dataset):
        shared_ids = np.instersect1d(train_dataset.index, test_dataset.index)
        self.assertTrue(len(shared_ids) == 0) 

    @parameterized.expand(splits)
    def test_all_data_covered_in_train_test_datasets(self, train_dataset, test_dataset):
        shared_ids = np.instersect1d(train_dataset.index, test_dataset.index)
        self.assertTrue(len(shared_ids) == len(TestClusterStratifiedShuffleSplit.dataset_df)) 

    # def test_approximately_even_label_distribution(self):
    #     pass


if __name__ == '__main__':

    unittest.main()

