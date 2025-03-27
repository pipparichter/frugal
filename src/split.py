from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedShuffleSplit
from src.dataset import Dataset
import json
import pandas as pd
import numpy as np


class ClusterStratifiedShuffleSplit():
    '''Implements a splitting strategy based on the results of clustering. The split ensures that all singleton clusters are 
    sorted into the training dataset during each split, and that all non-singleton clusters are homogenous.'''

    def __init__(self, dataset:Dataset, cluster_path:str=None, n_splits:int=5, test_size:float=0.2, train_size:float=0.8):
        
        self.dataset = dataset
        self._load_clusters(cluster_path)

        self.stratified_shuffle_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=42)

        self.adjusted_test_size = (test_size * self.n_non_singleton) / len(dataset)
        self.adjusted_train_size = (train_size * self.n_non_singleton + self.n_singleton) / len(dataset)
        self.train_size = train_size 
        self.test_size = test_size
        print(f'ClusterStratifiedShuffleSplit.__init__: Adjusted training and test sizes are {self.adjusted_train_size:.3f}, {self.adjusted_test_size:.3f}.')

        labels = self.cluster_df.cluster_label.values[self.non_singleton_idxs] # Stratify according to the cluster labels. 
        splits = self.stratified_shuffle_split.split(self.non_singleton_idxs, labels)
        # Need to map the split indices back over to the original dataset indices. 
        self.splits = [(self.non_singleton_idxs[train_idxs], self.non_singleton_idxs[test_idxs]) for train_idxs, test_idxs in splits]

        self.i = 0
        self.n_splits = n_splits 
        
        self._check()

    def _check(self):
        # Double check to make sure no singleton indices ended up in the split. 
        for train_idxs, test_idxs in self.splits:
            assert np.intersect1d(train_idxs, self.singleton_idxs).size == 0, 'ClusterStratifiedShuffleSplit._check: There are singleton indices in the split.'
            assert np.intersect1d(test_idxs, self.singleton_idxs).size == 0, 'ClusterStratifiedShuffleSplit._check: There are singleton indices in the split.'

    def _check_clusters(self, cluster_df):
        assert len(cluster_df) == len(self.dataset), 'ClusterStratifiedShuffleSplit._check_clusters: The dataset and cluster DataFrame indices do not match.'
        assert np.all(np.sort(cluster_df.index) == np.sort(self.dataset.index)), 'ClusterStratifiedShuffleSplit._check_clusters: The dataset and cluster DataFrame indices do not match.'

    @staticmethod
    def _split_non_homogenous_clusters(cluster_df:pd.DataFrame) -> pd.DataFrame:

        is_non_homogenous = lambda df : (df.label.nunique() > 1)
        is_homogenous = lambda df : (df.label.nunique() == 1)

        cluster_labels = cluster_df.cluster_label.unique()
        non_homogenous_cluster_labels = cluster_labels[cluster_df.groupby('cluster_label', sort=False).apply(is_non_homogenous, include_groups=False)]
        print(f'ClusterStratifiedShuffleSplit._split_non_homogenous_clusters: Found {len(non_homogenous_cluster_labels)} non-homogenous clusters.')

        max_cluster_label = cluster_labels.max()
        for cluster_label in non_homogenous_cluster_labels:
            cluster_ids = cluster_df[(cluster_df.cluster_label == cluster_label) & (cluster_df.label == 1)].index 
            cluster_df.loc[cluster_ids, 'cluster_label'] = max_cluster_label + 1
            max_cluster_label += 1

        assert np.all(cluster_df.groupby('cluster_label').apply(is_homogenous, include_groups=False)), f'ClusterStratifiedShuffleSplit._split_non_homogenous_clusters: There are still non-homogenous clusters.'
        return cluster_df

    def _load_clusters(self, path:str):

        cluster_df = pd.read_csv(path, index_col=0) # The index should be the sequence ID, and should have a cluster_label column. 
        self._check_clusters(cluster_df)
        cluster_df = cluster_df.loc[self.dataset.index].copy() # Make sure the index order matches. 
        cluster_df['label'] = self.dataset.label

        cluster_df = ClusterStratifiedShuffleSplit._split_non_homogenous_clusters(cluster_df)

        singleton = cluster_df.groupby('cluster_label', sort=False).apply(lambda df : (len(df) == 1), include_groups=False)
        cluster_df['singleton'] = cluster_df.cluster_label.map(singleton)
        self.cluster_df = cluster_df 
        self.singleton_idxs = np.where(cluster_df.singleton.values)[0]
        self.non_singleton_idxs = np.where(~cluster_df.singleton.values)[0]
        self.n_singleton = len(self.singleton_idxs)
        self.n_non_singleton = len(self.non_singleton_idxs)
        
        singleton_labels = self.dataset.label[self.singleton_idxs]
        # print(f'ClusterStratifiedShuffleSplit._load_clusters: Found {self.n_singleton} singleton clusters.')
        print(f'ClusterStratifiedShuffleSplit._load_clusters: Found {(singleton_labels == 1).sum()} singleton clusters with "real" labels.')
        print(f'ClusterStratifiedShuffleSplit._load_clusters: Found {(singleton_labels == 0).sum()} singleton clusters with "spurious" labels.')

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
        train_dataset = self.dataset.subset(train_idxs)
        test_dataset = self.dataset.subset(test_idxs) 

        train_dataset.set_attr('cluster_label', self.cluster_df.cluster_label.iloc[train_idxs])
        test_dataset.set_attr('cluster_label', self.cluster_df.cluster_label.iloc[test_idxs])

        return train_dataset, test_dataset
    
    def save(self, path:str, best_split:int=None):  
        content = dict()
        # Make sure everything is in the form of normal integers so it's JSON-serializable (not Numpy datatypes).
        for i, (train_idxs, test_idxs) in enumerate(self.splits):
            train_idxs = [int(idx) for idx in train_idxs]
            test_idxs = [int(idx) for idx in test_idxs]
            content[i] = {'train_idxs':list(train_idxs), 'test_idxs':list(test_idxs)}
        content['best_split'] = int(best_split)
        with open(path, 'w') as f:
            json.dump(content, f)

