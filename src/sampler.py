import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import WeightedRandomSampler
from numpy.random import choice
from scipy.special import softmax


class Sampler():

    def __init__(self, dataset, batch_size:int=16, balance_classes:bool=True, balance_lengths:bool=False, sample_size:int=None, **kwargs):

        self.labels = dataset.labels.numpy()
        self.weights = np.ones(len(dataset))
        self.n_per_class = [(self.labels == i).sum() for i in range(dataset.n_classes)] # The number of elements in each class. 
        self.n_total = len(dataset)
        self.n_classes = dataset.n_classes
        self.seqs = dataset.seqs

        if balance_classes:
            self._balance_classes()
        if balance_lengths:
            self._balance_lengths(**kwargs)
        
        self.idxs = np.arange(len(dataset))
        self.batch_size = batch_size
        self.n_batches = sample_size // batch_size + 1
        self.sample_size = (len(dataset) * self.n_classes) if (sample_size is None) else sample_size

        # self.sampler = torch.utils.data.WeightedRandomSampler(self.weights, self.sample_size, replacement=True)
        
    def _balance_classes(self):
        weights_per_class = np.array([(1 / (n_i)) for n_i in self.n_per_class])
        self.weights *= weights_per_class[self.labels]

    def _balance_lengths(self, n_bins:int=50, ref_class:int=1):
        lengths = np.array([len(seq) for seq in self.seqs])
        ref_lengths = lengths[self.labels == ref_class] 

        densities, bin_edges = np.histogram(ref_lengths, bins=n_bins, density=True)
        bin_min, bin_max = min(bin_edges), max(bin_edges) 
        densities = np.concatenate(([0], densities, [0]))

        bin_assignments = np.digitize(lengths, bin_edges)
        weights = densities[bin_assignments] # Get the weights, which is equivalent to the bin fraction in the reference class. 
        # self.weights *= weights 
        # This seems to work better if the weights are only applied to the non-reference class. 
        self.weights = np.where(self.labels != ref_class, weights * self.weights, self.weights)


    def __iter__(self):
        for _ in range(self.n_batches):
            yield choice(self.idxs, self.batch_size, replace=False, p=softmax(self.weights))

    def __len__(self):
        return self.sample_size

    

# def get_dataloader(dataset:Dataset, batch_size:int=16, balance_batches:bool=False) -> DataLoader:
#     '''Produce a DataLoader object for each batching of the input Dataset.'''
#     if balance_batches:
#         return DataLoader(dataset, sampler=dataset.sampler(), batch_size=batch_size)
#         # return torch.utils.data.DataLoader(dataset, batch_sampler= BalancedBatchSampler(dataset, batch_size=batch_size))
#     else:
#         return DataLoader(dataset, batch_size=batch_size, shuffle=True)
