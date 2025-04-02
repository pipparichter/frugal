import pandas as pd
import numpy as np
import torch
from torch.nn.functional import one_hot
from collections import namedtuple
import copy
import tables 
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def update_metadata(path:str, col:pd.Series):
    store = pd.HDFStore(path, 'a', table=True)
    metadata_df = store.get('metadata').copy() 
    try:
        assert len(col) == len(metadata_df), 'update_metadata: Index of the stored metadata and the column being added are unequal lengths.'
        assert len(np.intersect1d(metadata_df.index, col.index)) == len(metadata_df), 'update_metadata: Index of the stored metadata and the column being added are do not contain the same values.'

        col = col.loc[metadata_df.index] # Make sure the ordering is the same as in the stored metadata. 
        metadata_df[col.name] = col 
        store.put('metadata', metadata_df, format='table')
        print(f'update_metadata: Successfully added column {col.name} to the metadata.')
    except AssertionError as err:
        print(f'update_metadata: Failed with error "{err}". Closing file {path}')
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

    def __init__(self, embedding:np.ndarray=None, index:np.ndarray=None, scaled:bool=False, feature_type:str=None, path:str=None, **kwargs):

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
            self._label = torch.from_numpy(self.label.copy()).type(torch.LongTensor)
            self._label_one_hot_encoded = one_hot(self._label, num_classes=self.n_classes).to(torch.float32).to(DEVICE)

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
    def from_hdf(cls, path:str, feature_type:str=None, attrs:list=[]):
        embedding_df = Dataset._read_hdf(path, key=feature_type, chunk_size=100)
        metadata_df = Dataset._read_hdf(path, key='metadata')

        assert len(embedding_df) == len(metadata_df), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'
        assert np.all(embedding_df.index == metadata_df.index), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'

        index = embedding_df.index.values.copy() # Make sure this is a numpy array. 
        embedding = embedding_df.values.copy() # Why do I need to copy this?

        attrs = list(metadata_df.columns) if (attrs is None) else attrs
        kwargs = {attr:metadata_df[attr].values.copy() for attr in attrs}
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

    def subset(self, idxs):
        embedding = self.numpy()[idxs, :].copy() if self.has_embedding() else None
        index = self.index.copy()[idxs].copy()
        kwargs = {attr:getattr(self, attr)[idxs].copy() for attr in self.attrs}
        return Dataset(embedding, index=index, scaled=self.scaled, feature_type=self.feature_type, **kwargs)
    
    def to_hdf(self, path:str):
        store = pd.HDFStore(path, mode='w')

        metadata_df = self.metadata()
        store.put('metadata', metadata_df, format='table')
        
        if self.has_embedding():
            embedding_df = pd.DataFrame(self.numpy, index=pd.Series(self.index, name='id'))
            assert len(embedding_df) == len(metadata_df), 'Dataset.write: The indices of the embedding and the metadata do not match.'
            assert np.all(embedding_df.index == metadata_df.index), 'Dataset.write: The indices of the embedding and the metadata do not match.'
            store.put(self.feature_type, embedding_df, format='table')

        store.close()


class Pruner():

    def __init__(self, radius:float=2):
        
        self.radius = radius # Radius is inclusive. 
        self.graph = None 
        self.neighbor_idxs = None
        self.remove_ids = None
        self.keep_ids = None
        self.remove_idxs = None
        self.keep_idxs = None

    def _get_row(self, i:int):
        return np.array([self.graph[i, j] for j in self.neighbor_idxs[i]])
    
    def _adjust_graph_weights(self, dataset):

        for i in range(len(dataset)):
            self.graph[i, i] = np.nan # Don't want to consider distance to self. 

        # Don't want to prune nodes which are radius neighbors, but have opposite labels. 
        row_idxs = np.repeat(np.arange(self.graph.shape[0]), np.diff(self.graph.indptr)) # Extract row indices from the graph and convert out of CSR format. 
        col_idxs = self.graph.indices # Extract column indices graph and convert. 
        row_labels = dataset.label[row_idxs]
        col_labels = dataset.label[col_idxs]
        mask = (row_labels != col_labels)
        if mask.sum() > 0:
            print(f'Pruner._adjust_graph_weights: Removing {mask.sum()} edges where the endpoint nodes do not have the same label.')
        for i, j in zip(row_idxs[mask], col_idxs[mask]):
            # Set distances of opposite-labeled neighboring nodes to be NaNs.
            self.graph[i, j] = np.nan
            self.graph[j, i] = np.nan

    def fit(self, dataset):

        embeddings = dataset.numpy() 
        embeddings = StandardScaler().fit_transform(embeddings) # Make sure the embeddings are scaled. 

        print(f'Pruner.fit: Fitting the NearestNeighbors object with radius {self.radius}.')
        nearest_neighbors = NearestNeighbors(metric='euclidean', radius=self.radius)
        nearest_neighbors.fit(embeddings)
        
        print(f'Pruner.fit: Building the radius neighborhs graph.')
        self.graph = nearest_neighbors.radius_neighbors_graph(X=embeddings, radius=self.radius, mode='distance', sort_results=True)
        self.neighbor_idxs = nearest_neighbors.radius_neighbors(embeddings, return_distance=False, radius=self.radius)
        n_neighbors = np.array([len(idxs) for idxs in self.neighbor_idxs])
        self._adjust_graph_weights(dataset)

        idxs = np.arange(len(dataset))[np.argsort(n_neighbors)][::-1]
        remove_idxs = []
        for i in tqdm(idxs, desc='Pruner.fit: Pruning radius neighbors graph.'):
            # Want to nullify every point in the graph which has an edge to the node represented by i. 
            values = self._get_row(i)
            if np.any(~np.isnan(values)):
                remove_idxs.append(i.item())
                for j in self.neighbor_idxs[i]:
                    self.graph[i, j] = np.nan # Use nan instead of zero so that identical sequences also get removed.
                    self.graph[j, i] = np.nan 
        remove_idxs = np.array(remove_idxs)

        assert np.all(np.isnan(self.graph.data.ravel())), 'Pruner.fit: There are still non-NaN elements in the graph.'
        
        self.remove_idxs = remove_idxs
        self.remove_ids = dataset.index[remove_idxs]
        self.keep_ids = dataset.index[~np.isin(dataset.index, self.remove_ids)]
        self.keep_idxs = np.array([idx for idx in range(len(dataset)) if (idx not in remove_idxs)])

    def prune(self, dataset):
        print(f'Pruner.prune: Removing {len(self.remove_ids)} sequences from the input Dataset.')
        print(f'Pruner.prune: {len(self.keep_ids)} sequences remaining.')
        return dataset.subset(self.keep_idxs)
    

    def update_metadata(self, col:pd.Series):
        assert (self.path is not None), 'Dataset.update_metadata: The Dataset has no stored path.'
        update_metadata(self.path, col)
        







# def update(path:str, key:str, df:pd.DataFrame):

#     store = pd.HDFStore(path)
#     existing_df = store.get(key) # Get the data currently stored in the DataFrame. 
#     assert len(df) == len(existing_df), f'update: The indices of the existing and update DataFrames stored in {key} do not match.'
#     assert np.all(df.index == existing_df.index), f'update: The indices of the existing and update DataFrames stored in {key} do not match.'
    
#     store.put(key, df, format='table')
#     store.close()
        


# def build(name:str, genome_ids:list, output_dir:str='../data', ref_dir:str='../data/ref', labels_dir='../data/labels', spurious_ids:list=None, version:str=None, max_length:int=2000):

#     suffix = f'_{version}' if (version is not None) else ''

#     print(f'build: Loading data from {len(genome_ids)} genomes.')

#     # Can't rely on the top_hit_genome_id column for the genome IDs, because if there is no hit it is not populated.
#     ref_paths = [os.path.join(ref_dir, f'{genome_id}_summary.csv') for genome_id in genome_ids]
#     ref_dtypes = {'top_hit_partial':str, 'query_partial':str, 'top_hit_translation_table':str, 'top_hit_codon_start':str}
#     ref_df = pd.concat([pd.read_csv(path, index_col=0, dtype=ref_dtypes).assign(genome_id=get_genome_id(path)) for path in ref_paths])
    
#     labels_paths = [os.path.join(labels_dir, f'{genome_id}_label.csv') for genome_id in genome_ids] 
#     labels_df = pd.concat([pd.read_csv(path, index_col=0) for path in labels_paths])

#     df = ref_df.merge(labels_df, left_index=True, right_index=True, validate='one_to_one')
#     df = df.rename(columns={'query_seq':'seq'}) # Need to do this for file writing, etc. to work correctly, 
#     df = df.drop(columns=['top_hit_homolog_id', 'top_hit_homolog_seq', 'pseudo'])

#     lengths = df.seq.apply(len)
#     print(f'Removing {(lengths >= max_length).sum()} sequences exceeding the maximum length of {max_length}')
#     df = df[lengths < max_length]

#     if spurious_ids is not None:
#         df.loc[spurious_ids, 'label'] = 'spurious' # Update the labels according to the new output. 

#     all_df = df.copy()

#     df = df[df.label != 'none'].copy() # Filter out all of the hypothetical proteins with only ab initio evidence. 
#     df['label'] = [0 if (label == 'spurious') else 1 for label in df.label] # Convert labels to integers. 
#     print(f'build: Loaded {len(df)} sequences, {(df.label == 0).sum()} labeled spurious and {(df.label == 1).sum()} labeled real.')

#     real_df, spurious_df = df[df.label == 1].copy(), df[df.label == 0].copy()
#     n_real = len(real_df)
#     # Cluster only the real sequences at 50 percent similarity in hopes of better balancing the classes. 
#     mmseqs = MMseqs()
#     real_df = mmseqs.cluster(real_df, job_name=name, sequence_identity=0.50, reps_only=True, overwrite=False)
#     print(f'build: Clustering at 50 percent similarity removed {n_real - len(real_df)} sequences.')
#     mmseqs.cleanup()

#     df = pd.concat([spurious_df, real_df], ignore_index=False)

#     gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     train_idxs, test_idxs = list(gss.split(df, groups=df.genome_id))[0]
#     train_df, test_df = df.iloc[train_idxs], df.iloc[test_idxs]
#     print(f'build: {(train_df.label == 0).sum()} negative instances and {(train_df.label == 1).sum()} positive instances in the training dataset.')
#     print(f'build: {(test_df.label == 0).sum()} negative instances and {(test_df.label == 1).sum()} positive instances in the testing dataset.')

#     all_df['in_test_dataset'] = all_df.index.isin(test_df.index)
#     all_df['in_train_dataset'] = all_df.index.isin(train_df.index)
    
#     train_df.to_csv(os.path.join(output_dir, f'{name}_dataset_train{suffix}.csv'))
#     test_df.to_csv(os.path.join(output_dir, f'{name}_dataset_test{suffix}.csv'))
#     all_df.to_csv(os.path.join(output_dir, f'{name}_dataset_{suffix}.csv'))

#     return train_df, test_df, all_df 
