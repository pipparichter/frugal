import pandas as pd
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch.nn.functional import one_hot
import copy

class Dataset(torch.utils.data.Dataset):

    def __init__(self, embeddings:np.ndarray, labels:np.ndarray=None, index:np.ndarray=None):

        self.n_classes = n_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.labels = labels
        if (self.labels is not None):
            self.n_classes = len(np.unique(labels))
            self.labels = torch.from_numpy(self.labels).type(torch.LongTensor)
            self.labels_one_hot_encoded = one_hot(self.labels, num_classes=n_classes).to(torch.float32).to(self.device)

        self.index = index
        self.embeddings = torch.from_numpy(df.values).to(self.device)
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
    def from_hdf(cls, path:str, feature_type:str=None, labeled:bool=True):
        embeddings_df = pd.read_hdf(path, key=feature_type)
        labels_df = pd.read_hdf(path, key='metadata')['label']

        assert len(embeddings_df) == len(labels_df), 'Dataset.from_hdf: The indices of the labels and embeddings do not match.'
        assert np.all(embeddings_df.index == labels_df.index), 'Dataset.from_hdf: The indices of the labels and embeddings do not match.'

        return cls(embeddings_df.values, labels=labels_df.values, index=embeddings_df.index)
    
    def shape(self):
        return self.embeddings.shape

    def __getitem__(self, idx:int) -> Dict:
        item = {'embedding':self.embeddings[idx], 'index':self.index[idx], 'idx':idx}
        if self.labels is not None: # Include the label if the Dataset is labeled.
            item['label'] = self.labels[idx]
            item['label_one_hot_encoded'] = self.labels_one_hot_encoded[idx]
        return item


def get_weighted_random_sampler(dataset:Dataset, p:float=0.99):

    labels = dataset.labels.numpy()
    N = len(labels) # Total number of things in the dataset. 
    n = [(labels == i).sum() for i in range(dataset.n_classes)] # The number of elements in each class. 
    # Compute the minimum number of samples such that each training instance will probably be included at least once.
    s = int(max([np.log(1 - p) / np.log(1 - 1 / n_i) for n_i in n])) * dataset.n_classes
    w = [(1 / (n_i)) for n_i in n] # Proportional to the inverse frequency of each class. 
    return torch.utils.data.WeightedRandomSampler([w[i] for i in labels], s, replacement=True)

