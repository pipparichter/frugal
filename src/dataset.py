import pandas as pd
import numpy as np
import torch

from torch.nn.functional import one_hot
from sklearn.model_selection import train_test_split
from collections import namedtuple
import copy
# from sklearn.model_selection import train_test_split

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

Datasets = namedtuple('Datasets', ['train', 'test'])


class Dataset(torch.utils.data.Dataset):

    def __init__(self, embeddings:np.ndarray, labels:np.ndarray=None, seqs:np.ndarray=None, index:np.ndarray=None, scaled:bool=False):

        self.labels = labels
        self.n_features = embeddings.shape[-1]

        if (self.labels is not None):
            self.n_classes = len(np.unique(labels)) # Infer the number of classes based on the labels. 
            self.labels = torch.from_numpy(self.labels).type(torch.LongTensor)
            self.labels_one_hot_encoded = one_hot(self.labels, num_classes=self.n_classes).to(torch.float32).to(DEVICE)

        self.index = index
        self.embeddings = torch.from_numpy(embeddings).to(DEVICE)
        self.seqs = seqs
        self.n_features = self.embeddings.shape[-1]

        self.scaled = scaled
        self.length = len(embeddings)
        
    def __len__(self) -> int:
        return len(self.embeddings)


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
            seqs_df = pd.read_hdf(path, key='metadata')['seq']
            assert len(embeddings_df) == len(seqs_df), 'Dataset.from_hdf: The indices of the sequences and embeddings do not match.'
            assert np.all(embeddings_df.index == seqs_df.index), 'Dataset.from_hdf: The indices of the sequences and embeddings do not match.'
            seqs = seqs_df.values

        return cls(embeddings_df.values, labels=labels, seqs=seqs, index=embeddings_df.index)
    
    def shape(self):
        return self.embeddings.shape

    def __getitem__(self, idx:int) -> dict:
        item = {'embedding':self.embeddings[idx], 'index':self.index[idx]}
        if self.labels is not None: # Include the label if the Dataset is labeled.
            item['label'] = self.labels[idx]
            item['label_one_hot_encoded'] = self.labels_one_hot_encoded[idx]
        if self.seqs is not None:
            item['seq'] = self.seqs[idx]
        return item

    def subset(self, idxs):

        embeddings = self.embeddings.cpu().numpy()[idxs, :]
        labels = self.labels.cpu().numpy()[idxs] if (self.labels is not None) else self.labels
        seqs = self.seqs[idxs]
        index = self.index[idxs]
        scaled = self.scaled 

        return Dataset(embeddings, labels=labels, seqs=seqs, index=index, scaled=scaled)



def split(dataset:Dataset, test_size:float=0.2):

    idxs = np.arange(len(dataset))
    idxs_train, idxs_test = train_test_split(idxs, test_size=test_size)
    return dataset.subset(idxs_train), dataset.subset(idxs_test)