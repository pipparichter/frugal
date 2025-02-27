import pandas as pd 
import numpy as np 
from src import get_genome_id
import os 
from src.embed import get_embedder


class EmbeddingLibrary():
    '''This object is to make working with directories of PLM embeddings easier. Each directory holds a set of CSV files,
    where each file contains the embeddings for each gene in a particular microbial genome.'''
    feature_types = ['esm_650m_gap', 'esm_3b_gap', 'pt5_3b_gap']

    def __init__(self, dir_:str='../data/embeddings', feature_type:str='esm_650m_gap'):
        self.dir_ = os.path.join(dir_, feature_type)
        print(self.dir_)
        if not os.path.exists(self.dir_):
            print(f'EmbeddingLibrary.__init__: Creating library directory {self.dir_}.')
            os.makedirs(self.dir_) # Make the directory if it doesn't exist.

        self.feature_type = feature_type
        self.file_names = os.listdir(self.dir_)
        print(self.file_names)
        self.genome_ids = [get_genome_id(file_name) for file_name in self.file_names]
        self.file_name_map = {genome_id:file_name for genome_id, file_name in zip(self.genome_ids, self.file_names)}

        self.embedder = get_embedder(self.feature_type)

    def __len__(self):
        return len(self.genome_ids)

    def __copy__(self):
        return EmbeddingLibrary(dir_=os.path.dirname(self.dir_), feature_type=self.feature_type)

    def add(self, genome_id:str, df:pd.DataFrame):

        path = os.path.join(self.dir_, f'{genome_id}_embedding.csv')
        if os.path.exists(path):
            print(f'EmbeddingLibrary.add: File {path} already exists in the library.')
            return 

        embedder = get_embedder(self.feature_type)
        embeddings = self.embedder(df.seq.values.tolist())
        embeddings_df = pd.DataFrame(index=df.index)
        embeddings_df.to_csv(path)

    def get(self, genome_id:str, ids:list=None):
        
        path = os.path.join(self.dir_, f'{genome_id}_embedding.csv')
        embeddings_df = pd.read_csv(path, index_col=0)
        return embeddings_df.loc[ids, :].copy() if (ids is not None) else embeddings_df


def add(lib:EmbeddingLibrary, *file_names:list):
    # Expects the input directory to contain a bunch of FASTA protein files.
    for file_name in file_names:
        try:
            genome_id = get_genome_id(file_name)
            print(f'build_library: Generating embeddings for genome {genome_id}.')
            df = FASTAFile(path=os.path.join(args.input_dir, file_name)).to_df() # Don't need to parse the Prodigal output, as we just want the sequences.
            df = df[df.seq.apply(len) < 2000] # Filter out sequences which exceed the specified maximum length
            lib.add(genome_id, df)
        except Exception as err:
            print(f'build_library: Failed to generate embeddings for genome {genome_id}.')
            print(err)