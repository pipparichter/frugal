import pandas as pd
import numpy as np
import torch
import src 

from torch.nn.functional import one_hot
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from collections import namedtuple
import copy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

Datasets = namedtuple('Datasets', ['train', 'test'])

# https://stackoverflow.com/questions/53097952/how-to-understand-numpy-strides-for-layman

# TODO: Figure out why this error was happening.
# Traceback (most recent call last):
#   File "/central/groups/fischergroup/prichter/miniconda3/envs/tripy/bin/train", line 33, in <module>
#     sys.exit(load_entry_point('tripy', 'console_scripts', 'train')())
#              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
#   File "/central/groups/fischergroup/prichter/tripy/src/cli.py", line 180, in train
#     dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, load_seqs=True, load_labels=True)
#   File "/central/groups/fischergroup/prichter/tripy/src/dataset.py", line 52, in from_hdf
#     return cls(embeddings_df.values, labels=labels, seqs=seqs, index=embeddings_df.index.values, feature_type=feature_type, scaled=False)
#   File "/central/groups/fischergroup/prichter/tripy/src/dataset.py", line 30, in __init__
#     self.embeddings = torch.from_numpy(embeddings).to(DEVICE)
#                       ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
# ValueError: given numpy array strides not a multiple of the element byte size. Copy the numpy array to reallocate the memory.

class Dataset(torch.utils.data.Dataset):

    def __init__(self, embeddings:np.ndarray, metadata:pd.DataFrame=None, labels:np.ndarray=None, seqs:np.ndarray=None, index:np.ndarray=None, scaled:bool=False, feature_type:str=None):

        self.labels = labels
        self.n_features = embeddings.shape[-1]
        self.feature_type = feature_type 
        self.metadata = metadata 

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
    def from_hdf(cls, path:str, feature_type:str=None, load_seqs:bool=True, load_labels:bool=True, load_metadata:bool=True):
        embeddings_df = pd.read_hdf(path, key=feature_type)
        metadata_df = pd.read_hdf(path, key='metadata')
        
        assert len(embeddings_df) == len(metadata_df), 'Dataset.from_hdf: The indices of the embeddings and the metadata do not match.'
        assert np.all(embeddings_df.index == metadata_df.index), 'Dataset.from_hdf: The indices of the embeddings and the metadata do not match.'

        kwargs = dict()
        kwargs['labels'] = metadata_df.label.values if load_labels else None
        kwargs['seqs'] = metadata_df.seq.values if load_seqs else None
        kwargs['metadata'] = metadata_df if load_metadata else None
        kwargs['index'] = metadata_df.index.values

        return cls(embeddings_df.values.copy(), feature_type=feature_type, scaled=False, **kwargs)
    
    def shape(self):
        return self.embeddings.shape

    def __getitem__(self, idx:int) -> dict:
        item = {'embedding':self.embeddings[idx], 'idx':idx} # , 'index':[self.index[idx]]}
        if self.labels is not None: # Include the label if the Dataset is labeled.
            item['label'] = self.labels[idx]
            item['label_one_hot_encoded'] = self.labels_one_hot_encoded[idx]
        # if self.seqs is not None:
            # item['seq'] = [self.seqs[idx]]
        return item

    def subset(self, idxs):

        embeddings = self.embeddings.cpu().numpy()[idxs, :].copy()  
        labels = self.labels.cpu().numpy()[idxs] if (self.labels is not None) else None
        metadata = self.metadata.iloc[idxs].copy() if (self.metadata is not None) else None
        seqs = self.seqs[idxs].copy()
        index = self.index[idxs].copy()
        scaled = self.scaled 

        return Dataset(embeddings, labels=labels, seqs=seqs, index=index, scaled=scaled, feature_type=self.feature_type)



def split(dataset:Dataset, test_size:float=0.2, by:bool='genome_id'):

    idxs = np.arange(len(dataset))

    if by is not None:
        groups = dataset.metadata[by].values # Extract the values to split by. 
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idxs, test_idxs = list(gss.split(idxs, groups=groups))[0]
    else:
        idxs_train, idxs_test = train_test_split(idxs, test_size=test_size)
    
    return dataset.subset(idxs_train), dataset.subset(idxs_test)