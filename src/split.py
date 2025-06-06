from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedShuffleSplit
from src.dataset import Dataset
import json
import pandas as pd
import numpy as np
import os

# TODO: Might be nice to clean up the way cluster labels are managed, like store them as attributes in the 
#   Dataset, and verify that the cluster label is poplated before proceeding with the split.


class ClusterStratifiedShuffleSplit():
    '''Implements a splitting strategy based on the results of clustering. The split ensures that all singleton clusters are 
    sorted into the training dataset during each split, and that all non-singleton clusters are homogenous.'''

    def __init__(self, dataset:Dataset, n_splits:int=5, test_size:float=0.2, train_size:float=0.8):
        
        self.dataset = dataset
        self._load_clusters()

        cluster_ids = self.cluster_df.cluster_id.values[self.non_singleton_idxs] # Stratify non-singletons according to the cluster labels. 
        min_test_size = len(np.unique(cluster_ids)) # Test size cannot be smaller than the number of non-singleton clusters. 
        test_size = max(min_test_size, int(self.n_non_singleton * test_size))
        train_size = self.n_non_singleton - test_size # Does not include singletons. 

        self.stratified_shuffle_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=42)
        splits = self.stratified_shuffle_split.split(self.non_singleton_idxs, cluster_ids)
        # Need to map the split indices back over to the original dataset indices. Splits only contain non-singletons. 
        self.splits = [(self.non_singleton_idxs[train_idxs], self.non_singleton_idxs[test_idxs]) for train_idxs, test_idxs in splits]

        self.test_size = test_size
        self.train_size = self.n_singleton + train_size
        
        assert (self.train_size + self.test_size) == len(dataset), f'ClusterStratifiedShuffleSplit.__init__: The train and test sizes ({self.train_size}, {self.test_size}) do not equal the Dataset size ({len(dataset)}).'

        print(f'ClusterStratifiedShuffleSplit.__init__: Adjusted training set fraction is {self.train_size / len(dataset):.2f}.')
        print(f'ClusterStratifiedShuffleSplit.__init__: Adjusted testing set fraction is {self.test_size / len(dataset):.2f}.')

        self.i = 0
        self.n_splits = n_splits 
        
    @staticmethod
    def _check_homogenous_clusters(cluster_df:pd.DataFrame):
        '''Verify that all clusters are homogenous, i.e. every element in the cluster is assigned the same label.'''
        for cluster_id, df in cluster_df.groupby('cluster_id'):
            assert df.label.nunique() == 1, f'ClusterStratifiedShuffleSplit._check_homogenous_clusters: Cluster {cluster_id} is not homogenous.'
        print(f'ClusterStratifiedShuffleSplit._check_homogenous_clusters: All clusters in loaded file are homogenous.')

    def _load_clusters(self):

        assert self.dataset.clustered(), 'ClusterStratifiedShuffleSplit: The Dataset does not have associated cluster IDs.'
        assert self.dataset.labeled(), 'ClusterStratifiedShuffleSplit: The Dataset does not have associated labels.'

        cluster_df = self.dataset.metadata(attrs=['cluster_id', 'label'])

        # cluster_df = ClusterStratifiedShuffleSplit._split_non_homogenous_clusters(cluster_df)
        ClusterStratifiedShuffleSplit._check_homogenous_clusters(cluster_df)

        singleton = cluster_df.groupby('cluster_id', sort=False).apply(lambda df : (len(df) == 1), include_groups=False)
        cluster_df['singleton'] = cluster_df.cluster_id.map(singleton)

        self.cluster_df = cluster_df 
        self.singleton_idxs = np.where(cluster_df.singleton.values)[0]
        self.non_singleton_idxs = np.where(~cluster_df.singleton.values)[0]
        self.n_singleton = len(self.singleton_idxs)
        self.n_non_singleton = len(self.non_singleton_idxs)

        singleton_labels = cluster_df.label[cluster_df.singleton]
        # print(f'ClusterStratifiedShuffleSplit._load_clusters: Found {(singleton_labels == 1).sum()} singleton clusters with label 1.')
        # print(f'ClusterStratifiedShuffleSplit._load_clusters: Found {(singleton_labels == 0).sum()} singleton clusters with label 0.')

    def __len__(self):
        return self.n_splits

    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.i >= self.n_splits:
            raise StopIteration
        
        train_idxs, test_idxs = self.splits[self.i]
        train_idxs = np.concat([train_idxs, self.singleton_idxs], axis=None)

        self.i += 1 # Increment the counter.
        train_dataset = self.dataset.iloc(train_idxs)
        test_dataset = self.dataset.iloc(test_idxs) 

        return train_dataset, test_dataset
    


    # def _check(self):
    #     # Double check to make sure no singleton indices ended up in the split. 
    #     for train_idxs, test_idxs in self.splits:
    #         assert np.intersect1d(train_idxs, self.singleton_idxs).size == 0, 'ClusterStratifiedShuffleSplit._check: There are singleton indices in the split.'
    #         assert np.intersect1d(test_idxs, self.singleton_idxs).size == 0, 'ClusterStratifiedShuffleSplit._check: There are singleton indices in the split.'


