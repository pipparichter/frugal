import numpy as np 
import pandas as pd
from tqdm import tqdm 
import torch
from torch.utils.data import WeightedRandomSampler
from numpy.random import choice
from random import choices 
from scipy.special import softmax
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings 

# TODO: Is the sample size the correct size to return for the length method? Yes! 
# TODO: Look into why softmax doesn't seem to be behaving the way I expected. 

class Sampler():

    def __init__(self, dataset, batch_size:int=16, balance_classes:bool=False, balance_lengths:bool=False, sample_size:int=None, **kwargs):

        self.labels = dataset.labels.numpy()
        self.n_total = len(dataset)
        self.n_classes = dataset.n_classes
        self.seqs = dataset.seqs
        self.lengths = np.array([len(seq) for seq in self.seqs])
        self.idxs = np.arange(len(dataset))

        self.ref_class = kwargs.get('ref_class', 1 if balance_lengths else None)

        self.balance_lengths, self.balance_classes = balance_lengths, balance_classes
        self.length_weights = np.ones(len(self.idxs)) if (not balance_lengths) else self.get_length_weights(**kwargs)
        

        if balance_classes:
            # There are two options for balancing batches: randomly select the proportion of each batch which is made up of each class, 
            # weighting classes according to inverse frequency, or ensuring each batch is split evenly. The latter approach has the 
            # advantage of being faster. 
            
            # self.batch_size = batch_size
            # self.class_weights = self.get_class_weights()
            self.class_n_per_batch = batch_size // self.n_classes
            self.batch_size = self.class_n_per_batch * self.n_classes 
            if self.batch_size != batch_size:
                warnings.warn(f'Sampler.__init__: Specified batch size {batch_size} is not divisible by {self.n_classes}. Using batch size {self.batch_size} instead.')
        else:
            self.class_weights = np.ones(len(self.idxs))
            self.batch_size = batch_size

        self.sample_size = len(dataset) * self.n_classes if (sample_size is None) else sample_size
        self.n_batches = self.sample_size // batch_size + 1

        self.batches = self.get_batches()
        coverage = ', '.join([f'({i}) {100 * self.coverage(i):.2f}%' for i in range(self.n_classes)])
        print(f'Sampler.__init__: Generated {self.n_batches} batches of size {self.batch_size}. Coverage per class is {coverage}.')

    # def get_class_weights(self):
    #     n_per_class = [(self.labels == i).sum() for i in range(self.n_classes)] # The number of elements in each class.
    #     weights_per_class = np.array([(1 / (n_i)) for n_i in n_per_class])
    #     weights_per_class /= weights_per_class.max() # Normalize so it's on the same scale as the balance length weights. 
    #     return weights_per_class[self.labels] # Assign the weights using the original labels. 

    def get_length_weights(self, n_bins:int=50):
        assert self.ref_class is not None, 'Sampler.get_length_weights: Cannot balance by sequence length unless a reference class is specified.'

        ref_lengths = self.lengths[self.labels == self.ref_class] 
        ref_density, bin_edges = np.histogram(ref_lengths, bins=n_bins, density=True)
        bin_assignments = np.digitize(self.lengths, bin_edges)
        return np.concatenate([[0], ref_density, [0]])[bin_assignments]

    def nunique(self) -> int:
        batches = np.concatenate(self.batches)
        n_unique = len(np.unique(batches))
        return n_unique

    def coverage(self, label:int=1):
        '''Get the fraction of the specified class which is covered by the sampler.'''
        idxs = np.unique(np.concatenate(self.batches)) # Get all unique indices covered by the sampler. 
        batch_labels = self.labels[idxs]
        batch_n = (batch_labels == label).sum()
        total_n = (self.labels == label).sum()
        return (batch_n / total_n).item()

    def __iter__(self):
        for i in range(self.n_batches):
            yield self.batches[i]

    def get_batches(self):
        # Pre-sort the indices and weights to avoid repeated filtering (e.g. self.idxs[self.labels == i]).
        idxs = {i:self.idxs[self.labels == i] for i in range(self.n_classes)}
        length_weights = {i:self.length_weights[self.labels == i] for i in range(self.n_classes)}

        batches = [choices(idxs[i], k=self.class_n_per_batch * self.n_batches, weights=length_weights[i]) for i in range(self.n_classes)]
        batches = np.concatenate([np.split(np.array(idxs_), self.n_batches) for idxs_ in batches], axis=1)

        # batches = []
        # for _ in tqdm(range(self.n_batches), desc='Sampler.get_batches: Generating batches.'):
        #     batch_labels = np.array(choices(self.labels, k=self.batch_size, weights=self.class_weights))
        #     batch_n_per_class = np.bincount(batch_labels)
        #     batch_idxs = [idx for i, n in enumerate(batch_n_per_class) for idx in choices(idxs[i], k=n, weights=length_weights[i])]
        #     batches.append(np.array(batch_idxs))

        return batches

    def __len__(self):
        return self.sample_size

    def _plot_batch_classes(self, ax:plt.Axes, n_batches:int=20, color_map:dict={0:'tab:green', 1:'tab:red'}):

        batch_labels = [self.labels[batch] for batch in self.batches[:n_batches]]

        positions, bottom = np.arange(n_batches), np.zeros(n_batches)
        for i in range(self.n_classes):
            heights = np.array([(labels == i).sum() for labels in batch_labels])
            ax.bar(positions, heights, bottom=bottom, color=color_map[i], edgecolor='black', label=f'class {i}')
            bottom += heights 
        
        ax.legend()
        ax.set_ylabel('count')
        ax.set_xlabel('bin')

    def _plot_batch_lengths(self, ax:plt.Axes, n_batches:int=20, color_map:dict={0:'tab:green', 1:'tab:red'}):

        batch_labels = [self.labels[batch] for batch in self.batches[:n_batches]]
        batch_lengths = [self.lengths[batch] for batch in self.batches[:n_batches]]
        
        if (self.ref_class is not None):
            sns.kdeplot(self.lengths[self.labels == self.ref_class], ax=ax, color='black', label='ref.')

        for i, lengths, labels in zip(np.arange(n_batches), batch_lengths, batch_labels):
            for j in range(self.n_classes):
                sns.kdeplot(lengths[labels == j], ax=ax, color=color_map[j], label=(f'class {j}' if (i == 0) else None), lw=0.5, alpha=0.5)
        
        ax.legend()
        ax.set_ylabel('density')
        ax.set_xlabel('length (aa)')

    def _plot_coverage(self, ax:plt.Axes, color_map:dict={0:'tab:green', 1:'tab:red'}):
        sizes = [(self.labels == i).sum() for i in range(self.n_classes)]
        coverages = [self.coverage(i) for i in range(self.n_classes)]

        bar_width = 1 / self.n_classes - 0.1
        for i in range(self.n_classes):
            p = int(coverages[i] * 100)
            ax.bar([i - bar_width / 2], [coverages[i] * sizes[i]], color=color_map[i], edgecolor='black', label=f'class {i} ({p}% coverage)', width=bar_width)
            ax.bar([i + bar_width / 2], [sizes[i]], color=color_map[i], edgecolor='black', hatch='///', width=bar_width)
        ax.legend()
        ax.set_ylabel('count')
        ax.set_xlabel('class')

    def plot(self, path:str=None, color_map:dict={0:'tab:green', 1:'tab:red'}, n_batches:int=20):
        '''Visualize the behavior of the classifier.'''

        fig, axes = plt.subplots(ncols=3, figsize=(12, 4), layout='tight')

        title = [f'balance_lengths={self.balance_lengths}']
        title += [f'balance_classes={self.balance_classes}']
        title += [f'sample_size={self.sample_size}']
        title += [f'len(dataset)={len(self.idxs)}']
        fig.suptitle(', '.join(title), fontsize='medium')

        self._plot_batch_classes(axes[0], color_map=color_map, n_batches=n_batches)
        self._plot_batch_lengths(axes[1], color_map=color_map, n_batches=n_batches)
        self._plot_coverage(axes[2], color_map=color_map)

        if path is not None:
            fig.savefig(path)

        plt.show()

