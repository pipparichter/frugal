import numpy as np 
import pandas as pd 
import torch
from torch.utils import WeightedRandomSampler


class Sampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size:int=16, balance_classes:bool=True, balance_lengths:bool=False, sample_size:int=None, **kwargs):

        self.weights = np.ones(len(dataset))
        self.n_per_class = [(labels == i).sum() for i in range(dataset.n_classes)] # The number of elements in each class. 
        self.n_total = len(dataset)
        self.n_classes = dataset.n_classes
        self.labels = dataset.labels.numpy()

        if balance_classes:
            self._balance_classes(dataset)
        if balance_lengths:
            self._balance_lengths(dataset, **kwargs)

        self.sample_size = (len(dataset) * self.n_classes) if (sample_size is None) else sample_size
        self.sampler = torch.utils.data.WeightedRandomSampler(self.weights, self.sample_size, replacement=True)
        
    def _balance_classes(self, dataset):
        self.weights *= np.array([(1 / (n_i)) for n_i in self.n_per_class])

    def _balance_lengths(self, dataset, n_bins:int=50, ref_class:int=1):
        lengths = np.array([len(seq) for seq in dataset.seqs])
        ref_lengths = lengths[labels == ref_class] 

        n_per_bin, bin_edges = np.hist(ref_lengths, bins=n_bins)
        bin_min, bin_max = min(bin_edges), max(bin_edges) 
        frac_per_bin = n_per_bin / n_per_bin.sum() # Will have size n_bins. bin_edges has size n_bins + 1.
        frac_per_bin = np.concatenate(([0], frac_per_bin, [0]))

        bin_assignments = np.digitize(lengths, bin_edges)
        weights = frac_per_bin[bin_assignments] # Get the weights, which is equivalent to the bin fraction in the reference class. 
        self.weights *= weights 

    def __iter__(self):
        for batch in self.sampler:
            yield batch

    def __len__(self):
        return self.n_total

    

# def get_dataloader(dataset:Dataset, batch_size:int=16, balance_batches:bool=False) -> DataLoader:
#     '''Produce a DataLoader object for each batching of the input Dataset.'''
#     if balance_batches:
#         return DataLoader(dataset, sampler=dataset.sampler(), batch_size=batch_size)
#         # return torch.utils.data.DataLoader(dataset, batch_sampler= BalancedBatchSampler(dataset, batch_size=batch_size))
#     else:
#         return DataLoader(dataset, batch_size=batch_size, shuffle=True)
