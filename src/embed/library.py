import pandas as pd 
import numpy as np 
from src import get_genome_id
from src.embed.embedders import get_embedder
from src.files import FASTAFile
import os 

class EmbeddingLibrary():
    '''This object is to make working with directories of PLM embeddings easier. Each directory holds a set of CSV files,
    where each file contains the embeddings for each gene in a particular FASTA file (usually for a genome).'''
    feature_types = ['esm_650m_gap', 'esm_3b_gap', 'pt5_3b_gap']

    @staticmethod
    def _get_entry_name(path:str):
        '''Extract the name of the EmbeddingLibrary entry from the path name.'''
        entry_name = os.path.basename(path)
        return entry_name.replace('_embedding.csv', '')

    def __init__(self, dir_:str='../data/embeddings', feature_type:str='esm_650m_gap', max_length:int=2000):
        self.dir_ = os.path.join(dir_, feature_type)
        self.max_length = max_length
        if not os.path.exists(self.dir_):
            print(f'EmbeddingLibrary.__init__: Creating library directory {self.dir_}.')
            os.makedirs(self.dir_) # Make the directory if it doesn't exist.

        self.feature_type = feature_type
        self.file_names = os.listdir(self.dir_)
        self.entry_names = [EmbeddingLibrary._get_entry_name(file_name) for file_name in self.file_names]
        self.file_name_map = dict(list(zip(self.entry_names, self.file_names)))
        self.embedder = get_embedder(self.feature_type)

    def __len__(self):
        return len(self.entry_names)

    def copy(self):
        return EmbeddingLibrary(dir_=os.path.dirname(self.dir_), feature_type=self.feature_type, max_length=self.max_length)

    def add(self, entry_name:str, df:pd.DataFrame):

        mode, header = 'w', True
        path = os.path.join(self.dir_, f'{entry_name}_embedding.csv')

        if os.path.exists(path):
            existing_ids = pd.read_csv(path, usecols=['id']).values.ravel()
            df = df[~df.index.isin(existing_ids)].copy()
            if len(df) == 0:
                print(f'EmbeddingLibrary.add: File {path} already exists in embedding library. No new embeddings to add to the file.')
                return 
            print(f'EmbeddingLibrary.add: File {path} already exists in embedding library. Adding {len(df)} new embeddings to the file.')
            mode, header = 'a', False # Switch to append mode, and don't write the header. 
        
        embeddings = self.embedder(df.seq.values.tolist()) # 
        embedding_df = pd.DataFrame(embeddings, index=df.index)
        embedding_df.index.name = 'id'
        embedding_df.to_csv(path, mode=mode, header=header)

    def get(self, entry_name:str, ids:list=None):
        
        path = os.path.join(self.dir_, f'{entry_name}_embedding.csv')
        embedding_df = pd.read_csv(path, index_col=0)
        return embedding_df.loc[ids, :].copy() if (ids is not None) else embedding_df


def add(lib:EmbeddingLibrary, *paths:list):
    # Expects the input directory to contain a bunch of FASTA protein files.
    for path in paths:
        entry_name  = get_genome_id(path, errors='ignore', default=os.path.basename(path).split('.')[0])
        try:
            print(f'add: Generating embeddings for {entry_name}.')
            df = FASTAFile(path=path).to_df(prodigal_output=False) # Don't need to parse the Prodigal output, as we just want the sequences.
            df = df[df.seq.apply(len) < lib.max_length] # Filter out sequences which exceed the specified maximum length
            lib.add(entry_name, df)
        except Exception as err:
            print(f'add: Failed to generate embeddings for {entry_name}. Returned error message "{err}"')