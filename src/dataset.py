import pandas as pd
import numpy as np
import torch
from torch.nn.functional import one_hot
from collections import namedtuple
import copy
import tables 
from tqdm import tqdm
import warnings 


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def update_metadata(path:str, cols:list=None):
    store = pd.HDFStore(path, 'a', table=True)
    metadata_df = store.get('metadata').copy() 

    for col in cols:
        try:
            assert len(col) == len(metadata_df), f'update_metadata: Index of the stored metadata and the column being added are unequal lengths. Metadata has length {len(metadata_df)} and column has length {len(col)}.'
            assert len(np.intersect1d(metadata_df.index, col.index)) == len(metadata_df), 'update_metadata: Index of the stored metadata and the column being added are do not contain the same values.'
            col = col.loc[metadata_df.index] # Make sure the ordering is the same as in the stored metadata. 
            metadata_df[col.name] = col 
            store.put('metadata', metadata_df, format='table')
            print(f'update_metadata: Successfully added column {col.name} to the metadata.', flush=True)
        except AssertionError as err:
            print(f'update_metadata: Failed with error "{err}". Closing file {path}', flush=True)
    store.close() # Make sure to close the file, even in the case of an error. 


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
    label_map = {'spurious':0, 'real':1}

    def __init__(self, embedding:np.ndarray=None, index:np.ndarray=None, scaled:bool=False, feature_type:str='esm_650m_gap', path:str=None, **kwargs):

        self.path = path # Store the path from which the Dataset was loaded. 

        self.embedding = torch.from_numpy(embedding).to(DEVICE) if (embedding is not None) else embedding
        self.n_features = embedding.shape[-1] if (embedding is not None) else None
        self.index = index
        self.feature_type = feature_type 
        self.scaled = scaled
        
        self.attrs = list(kwargs.keys())
        for attr, value in kwargs.items():
            value = np.array(list(value))
            setattr(self, attr, value)

        # I think that prepending an underscore to the attribute name makes the attribute inaccessible from outside the class. 
        if ('label' in self.attrs):
            self.n_classes = len(np.unique(self.label)) # Infer the number of classes based on the label. 
            try:
                self._label = torch.from_numpy(self.label.copy()).type(torch.LongTensor).to(DEVICE)
                self._label_one_hot_encoded = one_hot(self._label, num_classes=self.n_classes).to(torch.float32).to(DEVICE)
            except RuntimeError as err:
                warnings.warn(f'Dataset.__init__: Unable to one-hot encode labels. {err}')
                self._label = None
                self._label_one_hot_encoded = None

    def has_embedding(self) -> bool:
        return (self.embedding is not None)

    def __len__(self) -> int:
        return len(self.index)
    
    @staticmethod 
    def _get_n_rows_hdf(path:str, key:str=None) -> int:
        f = tables.open_file(path, 'r')
        n_rows = f.get_node(f'/{key}').table.nrows
        f.close()
        return n_rows

    @staticmethod 
    def _read_hdf(path:str, chunk_size:int=None, key:str='esm_650m_gap') -> pd.DataFrame:
        n_rows = Dataset._get_n_rows_hdf(path, key=key)
        if (chunk_size is not None) and (n_rows > 50000): # If chunk size is specified, load in chunks with a progress bar. 
            n_chunks = n_rows // chunk_size + 1
            pbar = tqdm(pd.read_hdf(path, key=key, chunksize=chunk_size), total=n_chunks, desc=f'Dataset._read_hdf: Reading HDF file from {path}')
            df = [chunk for chunk in pbar]
            df = pd.concat(df)
        else:
            df = pd.read_hdf(path, key=key)
        return df

    @classmethod
    def from_hdf(cls, path:str, feature_type:str='esm_650m_gap', attrs:list=[]):
        embedding_df = Dataset._read_hdf(path, key=feature_type, chunk_size=100)
        index = embedding_df.index.values.copy() # Make sure this is a numpy array. 
        embedding = embedding_df.values.copy() # Why do I need to copy this?

        try:
            metadata_df = Dataset._read_hdf(path, key='metadata')
            assert len(embedding_df) == len(metadata_df), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'
            assert np.all(embedding_df.index == metadata_df.index), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'
            attrs = list(metadata_df.columns) if (attrs is None) else attrs
            kwargs = {attr:metadata_df[attr].values.copy() for attr in attrs}
        except Exception: # TODO: I should make this more specific. 
            print(f'Dataset.from_hdf: No metadata stored in the Dataset at {path}')
            kwargs = dict()

        return cls(embedding, feature_type=feature_type, index=index, scaled=False, path=path, **kwargs)
    
    # @classmethod
    # def from_csv(cls, path:str, attrs:list=None):
    #     df = pd.read_csv(path, index_col=0)
    #     attrs = list(df.columns) if (attrs is None) else attrs
    #     kwargs = {attr:df[attr].values for attr in attrs}
    #     return cls(index=df.index.values, scaled=False, **kwargs)

    def shape(self):
        return self.embedding.shape if self.has_embedding() else self.index.shape
    
    def clustered(self):
        return hasattr(self, 'cluster_id')

    def labeled(self):
        return hasattr(self, 'label') 
    
    def numpy(self):
        '''Return the stored embeddings as a NumPy array. If no embeddings are stored, return None.'''
        return copy.deepcopy(self.embedding).cpu().numpy() if self.has_embedding() else None

    def metadata(self, attrs:list=None) -> pd.DataFrame:
        attrs = self.attrs if (attrs is None) else attrs
        metadata_df = {attr:getattr(self, attr) for attr in attrs}
        metadata_df = pd.DataFrame(metadata_df, index=pd.Series(self.index, name='id'))
        return metadata_df

    def __getitem__(self, idx:int) -> dict:
        item = {'embedding':self.embedding[idx], 'idx':idx} # , 'index':[self.index[idx]]}
        if hasattr(self, 'label'): # Include the label if the Dataset is labeled.
            item['label'] = self._label[idx]
            item['label_one_hot_encoded'] = self._label_one_hot_encoded[idx]
        return item

    def set_attr(self, attr:str, values:pd.Series):
        assert len(values) == len(self), 'Dataset.set_attr: The length of the attribute values does not match the length of the Dataset.'
        assert np.all(values.index == self.index), 'Dataset.set_attr: The index of the attribute values does not match the index of the Dataset.'
        values = values.loc[self.index] # Make sure the index in the series is the same as that of the DataFrame. 
        self.attrs.append(attr)
        setattr(self, attr, values.values)

    def iloc(self, idxs):
        '''Get a subset of the Dataset using numerical indices.'''
        embedding = self.numpy()[idxs, :].copy() if self.has_embedding() else None
        index = self.index.copy()[idxs].copy()
        kwargs = {attr:getattr(self, attr)[idxs].copy() for attr in self.attrs}
        return Dataset(embedding, index=index, scaled=self.scaled, feature_type=self.feature_type, **kwargs)
    
    def loc(self, ids):
        '''Get a subset of the Dataset using sequence IDs.'''
        idxs = np.where(np.isin(self.index, ids))[0]
        return self.iloc(idxs)
    
    def concat(self, dataset):
        assert not self.scaled, 'Dataset.concat: Cannot concatenate a scaled dataset.'
        assert not dataset.scaled, 'Dataset.concat: Cannot concatenate a scaled dataset.'
        embedding = np.concatenate([self.numpy(), dataset.numpy()], axis=0)
        index = np.concatenate([self.index, dataset.index]).ravel()
        return Dataset(embedding, index=index, scaled=False, feature_type=self.feature_type)
    
    def to_hdf(self, path:str):
        store = pd.HDFStore(path, mode='a', table=True)

        metadata_df = self.metadata()
        store.put('metadata', metadata_df, format='table')
        
        if self.has_embedding():
            embedding_df = pd.DataFrame(self.numpy(), index=pd.Series(self.index, name='id'))
            assert len(embedding_df) == len(metadata_df), 'Dataset.write: The indices of the embedding and the metadata do not match.'
            assert np.all(embedding_df.index == metadata_df.index), 'Dataset.write: The indices of the embedding and the metadata do not match.'
            store.put(self.feature_type, embedding_df, format='table')

        store.close()

