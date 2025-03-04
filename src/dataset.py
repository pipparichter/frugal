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
#     dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, load_seq=True, load_label=True)
#   File "/central/groups/fischergroup/prichter/tripy/src/dataset.py", line 52, in from_hdf
#     return cls(embedding_df.values, label=label, seq=seq, index=embedding_df.index.values, feature_type=feature_type, scaled=False)
#   File "/central/groups/fischergroup/prichter/tripy/src/dataset.py", line 30, in __init__
#     self.embedding = torch.from_numpy(embedding).to(DEVICE)
#                       ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
# ValueError: given numpy array strides not a multiple of the element byte size. Copy the numpy array to reallocate the memory.

class Dataset(torch.utils.data.Dataset):

    def __init__(self, embedding:np.ndarray, index:np.ndarray=None, scaled:bool=False, feature_type:str=None, **kwargs):

        self.embedding = torch.from_numpy(embedding).to(DEVICE)
        self.n_features = self.embedding.shape[-1]
        self.index = index.values
        self.feature_type = feature_type 
        self.scaled = scaled
        
        self.attrs = list(kwargs.keys())
        for attr, value in kwargs.items():
            setattr(self, attr, value.values)

        if ('label' in self.attrs):
            self.n_classes = len(np.unique(self.label)) # Infer the number of classes based on the label. 
            self.label = torch.from_numpy(self.label).type(torch.LongTensor)
            self.label_one_hot_encoded = one_hot(self.label, num_classes=self.n_classes).to(torch.float32).to(DEVICE)

    def __len__(self) -> int:
        return len(self.embedding)

    @classmethod
    def from_hdf(cls, path:str, feature_type:str=None, attrs:list=['genome_id', 'seq', 'label']):
        embedding_df = pd.read_hdf(path, key=feature_type)
        metadata_df = pd.read_hdf(path, key='metadata')
        
        assert len(embedding_df) == len(metadata_df), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'
        assert np.all(embedding_df.index == metadata_df.index), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'

        index = embedding_df.index.copy()
        embedding = embedding_df.values.copy() # Why do I need to copy this?

        kwargs = {attr:getattr(metadata_df, attr, None) for attr in attrs}
        kwargs = {attr:np.array(value) for attr, value in kwargs.items() if (value is not None)}
        return cls(embedding, feature_type=feature_type, index=index, scaled=False, **kwargs)
    
    def shape(self):
        return self.embedding.shape

    def __getitem__(self, idx:int) -> dict:
        item = {'embedding':self.embedding[idx], 'idx':idx} # , 'index':[self.index[idx]]}
        if hasattr(self, 'label'): # Include the label if the Dataset is labeled.
            item['label'] = self.label[idx]
            item['label_one_hot_encoded'] = self.label_one_hot_encoded[idx]
        return item

    def subset(self, idxs):
        embedding = self.embedding.cpu().numpy()[idxs, :].copy()  
        index = self.index.copy()
        kwargs = {attr:getattr(self, attr)[idxs].copy() for attr in self.attrs}
        return Dataset(embedding, index=index, scaled=self.scaled, feature_type=self.feature_type, **kwargs)



def split(dataset:Dataset, test_size:float=0.2, by:bool='genome_id'):

    idxs = np.arange(len(dataset))

    if by is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        idxs_train, idxs_test = list(gss.split(idxs, groups=getattr(dataset, by)))[0]
    else:
        idxs_train, idxs_test = train_test_split(idxs, test_size=test_size)
    
    return dataset.subset(idxs_train), dataset.subset(idxs_test)