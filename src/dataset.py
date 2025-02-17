import pandas as pd
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch.nn.functional import one_hot
import copy

class Dataset(torch.utils.data.Dataset):

    def __init__(self, embeddings:np.ndarray, labels:np.ndarray=None, seqs:np.ndarray=None, index:np.ndarray=None):

        self.n_classes = n_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.labels = labels
        if (self.labels is not None):
            self.n_classes = len(np.unique(labels))
            self.labels = torch.from_numpy(self.labels).type(torch.LongTensor)
            self.labels_one_hot_encoded = one_hot(self.labels, num_classes=n_classes).to(torch.float32).to(self.device)

        self.index = index
        self.embeddings = torch.from_numpy(df.values).to(self.device)
        self.seqs = seqs
        self.n_features = self.embeddings.shape[-1]

        self.scaled = False
        self.length = len(df)
        
    def __len__(self) -> int:
        return len(self.embeddings)


    def scale(self, scaler):
        # Repeatedly scaling a Dataset causes problems, though I am not sure why. I would have thought that
        # subsequent applications of a scaler would have no effect. 
        assert not self.scaled, 'Dataset.scale: The dataset has already been scaled.'
        
        embeddings = copy.deepcopy(self.embeddings.cpu().numpy())
        embeddings = scaler.transform(embeddings)
        embeddings = torch.FloatTensor(embeddings).to(dataset.device)
        
        labels = copy.deepcopy(self.labels.cpu().numpy()) if (self.labels is not None) else self.labels
        
        dataset = Dataset(embeddings, labels=labels)
        dataset.scaled = True
        return dataset

    @classmethod
    def from_hdf(cls, path:str, feature_type:str=None, load_seqs:bool=True, load_labels:bool=True):
        embeddings_df = pd.read_hdf(path, key=feature_type)
        labels, seqs = None, None

        if load_labels:
            labels_df = pd.read_hdf(path, key='metadata')['label']
            assert len(embeddings_df) == len(labels_df), 'Dataset.from_hdf: The indices of the labels and embeddings do not match.'
            assert np.all(embeddings_df.index == labels_df.index), 'Dataset.from_hdf: The indices of the labels and embeddings do not match.'
            labels = labels_df.values
        
        if load_seqs:
            seqs_df = pd.read_hdf(path, key='metadata')['label']
            assert len(embeddings_df) == len(seqs_df), 'Dataset.from_hdf: The indices of the sequences and embeddings do not match.'
            assert np.all(embeddings_df.index == seqs_df.index), 'Dataset.from_hdf: The indices of the sequences and embeddings do not match.'
            seqs = seqs_df.values

        return cls(embeddings_df.values, labels=labels, seqs=seqs, index=embeddings_df.index)
    
    def shape(self):
        return self.embeddings.shape

    def __getitem__(self, idx:int) -> Dict:
        item = {'embedding':self.embeddings[idx], 'index':self.index[idx], 'idx':idx}
        if self.labels is not None: # Include the label if the Dataset is labeled.
            item['label'] = self.labels[idx]
            item['label_one_hot_encoded'] = self.labels_one_hot_encoded[idx]
        return item


def get_balanced_class_sampler(dataset:Dataset, p:float=0.99):

    labels = dataset.labels.numpy()
    n_total = len(labels) # Total number of things in the dataset. 
    n_per_class = [(labels == i).sum() for i in range(dataset.n_classes)] # The number of elements in each class. 
    # Compute the minimum number of samples such that each training instance will probably be included at least once.
    sample_size = int(max([np.log(1 - p) / np.log(1 - 1 / n_i) for n_i in n_per_class])) * dataset.n_classes
    weights = [(1 / (n_i)) for n_i in n_per_class] # Proportional to the inverse frequency of each class. 
    return torch.utils.data.WeightedRandomSampler([weights[i] for i in labels], sample_size, replacement=True)


def get_balanced_length_sampler(dataset:Dataset, ref_class:int=1, n_bins:int=50):

    labels = dataset.labels.numpy()
    lengths = np.array([len(seq) for seq in dataset.seqs])

    n_per_class = [(labels == i).sum() for i in range(dataset.n_classes)] # The number of elements in each class. 
    ref_lengths = lengths[labels == ref_class] 

    n_per_bin, bin_edges = np.hist(ref_lengths, bins=n_bins)
    bin_min, bin_max = min(bin_edges), max(bin_edges) 
    frac_per_bin = n_per_bin / n_per_bin.sum() # Will have size n_bins. bin_edges has size n_bins + 1.
    frac_per_bin = np.concatenate(([0], frac_per_bin, [0]))

    bin_assignments = np.digitize(lengths, bin_edges)
    weights = frac_per_bin[bin_assignments] # Get the weights, which is equivalent to the bin fraction in the reference class. 

    sample_size = max(n_per_class) * dataset.n_classes # I might want to adjust this. 
    return torch.utils.data.WeightedRandomSampler(weights, sample_size, replacement=True)

