import pandas as pd
import numpy as np
import torch
from src import get_genome_id
import os
from torch.nn.functional import one_hot
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedShuffleSplit
from collections import namedtuple
import copy
import tables 
from tqdm import tqdm
import json
import pandas as pd 
import numpy as np 
# from sklearn.cluster import DBSCAN # , OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances



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
    label_map = {'spurious':0, 'real':1}

    def __init__(self, embedding:np.ndarray, index:np.ndarray=None, scaled:bool=False, feature_type:str=None, **kwargs):

        self.embedding = torch.from_numpy(embedding).to(DEVICE)
        self.n_features = self.embedding.shape[-1]
        self.index = index
        self.feature_type = feature_type 
        self.scaled = scaled
        
        self.attrs = list(kwargs.keys())
        for attr, value in kwargs.items():
            setattr(self, attr, value)

        # I think that prepending an underscore to the attribute name makes the attribute inaccessible from outside the class. 
        if ('label' in self.attrs):
            self.n_classes = len(np.unique(self.label)) # Infer the number of classes based on the label. 
            self._label = torch.from_numpy(self.label.copy()).type(torch.LongTensor)
            self._label_one_hot_encoded = one_hot(self._label, num_classes=self.n_classes).to(torch.float32).to(DEVICE)

    def __len__(self) -> int:
        return len(self.embedding)
    
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
    def from_hdf(cls, path:str, feature_type:str=None, attrs:list=['genome_id', 'seq', 'label']):
        embedding_df = Dataset._read_hdf(path, key=feature_type, chunk_size=100)
        metadata_df = Dataset._read_hdf(path, key='metadata')

        assert len(embedding_df) == len(metadata_df), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'
        assert np.all(embedding_df.index == metadata_df.index), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'

        index = embedding_df.index.values.copy() # Make sure this is a numpy array. 
        embedding = embedding_df.values.copy() # Why do I need to copy this?

        kwargs = {attr:getattr(metadata_df, attr, None) for attr in attrs}
        kwargs = {attr:np.array(value) for attr, value in kwargs.items() if (value is not None)}
        return cls(embedding, feature_type=feature_type, index=index, scaled=False, **kwargs)
    
    def shape(self):
        return self.embedding.shape
    
    def to_df(self) -> pd.DataFrame:
        embedding = copy.deepcopy(self.embedding).cpu().numpy()
        df = pd.DataFrame(embedding, index=self.index)
        df = df[~df.index.duplicated(keep='first')].copy()
        return df

    def __getitem__(self, idx:int) -> dict:
        item = {'embedding':self.embedding[idx], 'idx':idx} # , 'index':[self.index[idx]]}
        if hasattr(self, 'label'): # Include the label if the Dataset is labeled.
            item['label'] = self._label[idx]
            item['label_one_hot_encoded'] = self._label_one_hot_encoded[idx]
        return item

    def set_attr(self, attr:str, values:pd.Series):
        assert len(values) == len(self), 'Dataset.set_attr: The length of the attribute values does not match the length of the dataset.'
        values = values.loc[self.index] # Make sure the index in the series is the same as that of the DataFrame. 
        self.attrs.append(attr)
        setattr(self, attr, values.values)

    def subset(self, idxs):
        embedding = self.embedding.cpu().numpy()[idxs, :].copy()  
        index = self.index.copy()[idxs].copy()
        kwargs = {attr:getattr(self, attr)[idxs].copy() for attr in self.attrs}
        return Dataset(embedding, index=index, scaled=self.scaled, feature_type=self.feature_type, **kwargs)
    
    def write(self, path:str):
        metadata_df = {attr:getattr(self, attr) for attr in self.attrs}
        metadata_df = pd.DataFrame(metadata_df, index=self.index)
        embedding_df = pd.DataFrame(self.embedding.cpu().numpy(), index=self.index)

        assert len(embedding_df) == len(metadata_df), 'Dataset.write: The indices of the embedding and the metadata do not match.'
        assert np.all(embedding_df.index == metadata_df.index), 'Dataset.write: The indices of the embedding and the metadata do not match.'

        store = pd.HDFStore(path, mode='w')
        store.put('metadata', metadata_df, format='table')
        store.put(self.feature_type, embedding_df, format='table')
        store.close()


# I wonder if I should be smarter about how I split the dataset. 

class Splitter():

    def __init__(self, dataset:Dataset, n_splits:int=5, test_size:float=0.1, train_size:float=0.9):

        self.stratified_shuffle_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=42)
        idxs = np.arange(len(dataset))
        self.splits = list(self.stratified_shuffle_split.split(idxs, dataset.label))

        self.i = 0
        self.n_splits = n_splits 
        self.train_size = train_size 
        self.test_size = test_size
        self.dataset = dataset 

    def __len__(self):
        return self.n_splits

    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.i >= self.n_splits:
            raise StopIteration
        
        train_idxs, test_idxs = self.splits[self.i]
        # datasets = Datasets(self.dataset.subset(train_idxs), self.dataset.subset(test_idxs))

        self.i += 1 # Increment the counter. 
        return self.dataset.subset(train_idxs), self.dataset.subset(test_idxs) 
    
    def save(self, path:str, best_split:int=None):  
        content = dict()
        # Make sure everything is in the form of normal integers so it's JSON-serializable (not Numpy datatypes).
        for i, (train_idxs, test_idxs) in enumerate(self.splits):
            train_idxs = [int(idx) for idx in train_idxs]
            test_idxs = [int(idx) for idx in test_idxs]
            content[i] = {'train_idxs':list(train_idxs), 'test_idxs':list(test_idxs)}
        content['best_split'] = int(best_split)
        with open(path, 'w') as f:
            json.dump(content, f)



class Pruner():

    def __init__(self, radius:float=None):
        
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

        embeddings = dataset.embedding.to(torch.float16).numpy() # Use half precision to reduce memory. 

        print(f'Pruner.fit: Fitting the NearestNeighbors object with radius {self.radius}.')
        nearest_neighbors = NearestNeighbors(metric='minkowski', p=2, radius=self.radius)
        nearest_neighbors.fit(embeddings)
        
        print(f'Pruner.fit: Building the radius neighborhs graph.')
        self.graph = nearest_neighbors.radius_neighbors_graph(X=embeddings, radius=self.radius, mode='distance', sort_results=True)
        self.neighbor_idxs = nearest_neighbors.radius_neighbors(embeddings, return_distance=False, radius=self.radius)
        n_neighbors = np.array([len(idxs) for idxs in self.neighbor_idxs])
        self._adjust_graph_weights(dataset)

        # dists = pairwise_distances(embeddings, metric='minkowski', p=2)
        # expected_edges = np.sort(dists[dists <= self.radius].ravel())
        # expected_n_neighbors = (dists <= self.radius).sum(axis=1).ravel()
        # assert len(expected_edges) == len(edges), f'Pruner.fit: Expected {len(expected_edges)} edges, but the radius graph has {len(edges)} edges.'
        # assert np.all(np.round(edges, 3) == np.round(edges, 3)), f'Pruner.fit: The edge weights in the radius graph do not match the expected values.'
        # assert np.all(expected_n_neighbors == n_neighbors), f'Pruner.fit: The expected number of neighbors per node does not match the expected number of neighbors.'

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
        


def build(genome_ids:list, output_path:str='../data', ref_dir:str='../data/ref', labels_dir='../data/labels', max_length:int=2000, labeled:bool=True):

    print(f'build: Loading data from {len(genome_ids)} genomes.')

    # Can't rely on the top_hit_genome_id column for the genome IDs, because if there is no hit it is not populated.
    ref_paths = [os.path.join(ref_dir, f'{genome_id}_summary.csv') for genome_id in genome_ids]
    ref_dtypes = {'top_hit_partial':str, 'query_partial':str, 'top_hit_translation_table':str, 'top_hit_codon_start':str}
    ref_df = pd.concat([pd.read_csv(path, index_col=0, dtype=ref_dtypes).assign(genome_id=get_genome_id(path)) for path in ref_paths])
    
    labels_paths = [os.path.join(labels_dir, f'{genome_id}_label.csv') for genome_id in genome_ids] 
    labels_df = pd.concat([pd.read_csv(path, index_col=0) for path in labels_paths])

    df = ref_df.merge(labels_df, left_index=True, right_index=True, validate='one_to_one')
    df = df.rename(columns={'query_seq':'seq'}) # Need to do this for file writing, etc. to work correctly, 
    df = df.drop(columns=['top_hit_homolog_id', 'top_hit_homolog_seq', 'pseudo'])

    mask = df.seq.apply(len) < max_length
    print(f'build: Removing {(~mask).sum()} sequences exceeding the maximum length of {max_length}')
    df = df[mask].copy()

    if labeled:
        mask = (df.label != 'none')
        print(f'build: Removing {(~mask).sum()} which have not been assigned a label.')
        df = df[mask].copy()
        df['label'] = df.label.map({'spurious':0, 'real':1}) # Convert the remaining labels to integers. 
    
    df.to_csv(output_path)
        


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
