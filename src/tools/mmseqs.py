import pandas as pd 
import os 
from src.files import FASTAFile
import subprocess
import shutil
import numpy as np 
from scipy.sparse import csr_matrix, save_npz, lil_matrix
import subprocess
from tqdm import tqdm

# From the MMseqs user guide: The sequence identity is estimated from the local alignment bit score divided 
# by the maximum length of the two aligned sequence segments. A linear regression function which correlates this normalized
# score to sequence identity is then used to estimate the sequence identity. This is a better measure of degree of similarity
# than the actual sequence identity, because it also takes the degree of similarity between aligned amino acids and the number and length
# of gaps into account

def alignment_to_csr_matrix(path:str, output_path:str=None, index=np.ndarray, chunk_size:int=100):

    output_path = path.replace('.tsv', '.npz') if (output_path is None) else output_path
    # matrix = csr_matrix((len(index), len(index)), dtype=np.float32)
    matrix = lil_matrix((len(index), len(index)), dtype=np.float32)
    idxs = {id_:idx for idx, id_ in enumerate(index)}
    
    n_alignments = int(subprocess.run(f'wc -l {path}', shell=True, check=True, capture_output=True).stdout.split()[0])
    chunks = MMSeqs.load_align(path, chunk_size=chunk_size)

    for chunk in tqdm(chunks, total=n_alignments // chunk_size + 1, desc='alignment_to_csr_matrix'):
        chunk = chunk[chunk.query_id != chunk.subject_id] # Remove self-alignments. 

        for row in chunk.itertuples():
            i, j = idxs[row.query_id], idxs[row.subject_id]
            matrix[i, j] = max(row.sequence_identity, matrix[i, j])
            matrix[j, i] = max(row.sequence_identity, matrix[i, j])
    save_npz(output_path, matrix.tocsr(), compressed=True)
    return matrix


class MMSeqs():

    cleanup_files = ['{name}_rep_seq.fasta', '{name}_all_seqs.fasta']

    align_fields = ['query_id', 'subject_id', 'sequence_identity', 'alignment_length', 'n_mismatches', 'n_gaps']
    align_fields += ['query_alignment_start', 'query_alignment_stop', 'subject_alignment_start', 'subject_alignment_stop']
    align_fields += ['e_value', 'bit_score']

    cluster_fields = ['cluster_rep', 'id']

    modules = ['cluster', 'align']
    prefix = 'mmseqs'

    def __init__(self, dir_:str='../data/mmseqs'):

        # Need a directory to store temporary files. If one does not already exist, create it in the working directory.
        self.tmp_dir = os.path.join(dir_, 'tmp') 
        self.dir_ = dir_ 

        if not os.path.exists(self.dir_):
            os.mkdir(self.dir_)
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
        self.cleanup_files = []

    def cleanup(self):
        for path in self.cleanup_files:
            if os.path.exists(path):
                os.remove(path)

    def _make_database_dir(self, database_name:str):
        database_dir = os.path.join(self.dir_, database_name)
        # Making a database produces a lot of output files, so want to organize them into directories. 
        if not os.path.exists(database_dir):
            os.mkdir(database_dir)
        return database_dir

    def _make_database(self, df:pd.DataFrame, name:str=None):
        '''Create an mmseqs database from a FASTA file, using the sequences in the input DataFrame.'''
        database_name = f'{name}_database'
        database_dir = self._make_database_dir(database_name)
        database_path = os.path.join(database_dir, database_name)

        input_path = os.path.join(self.dir_, name + '.faa')

        print(f'MMSeqs._make_database: Creating database {database_name} in {database_dir}')
        FASTAFile(df=df).write(input_path)
        subprocess.run(f'mmseqs createdb {input_path} {database_path}', shell=True, check=True, stdout=subprocess.DEVNULL)

        return database_path 

    def _prefilter(self, query_df:pd.DataFrame, subject_df:pd.DataFrame=None, query_name:str=None, subject_name:str=None, sensitivity:float=None):

        subject_name = query_name if (subject_name is None) else subject_name

        output_database_name = f'{query_name}_{subject_name}_prefilter_database'
        output_database_dir = self._make_database_dir(output_database_name)
        output_database_path = os.path.join(output_database_dir, output_database_name)

        if subject_df is not None:
            query_database_path = self._make_database(query_df, name=query_name)
            subject_database_path = self._make_database(subject_df, name=subject_name)
        else:
            query_database_path = self._make_database(query_df, name=query_name)
            subject_database_path = query_database_path

        cmd = f'mmseqs prefilter {query_database_path} {subject_database_path} {output_database_path}'
        if sensitivity is not None:
            cmd += f' -s {sensitivity}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        
        return query_database_path, subject_database_path, output_database_path
    
    # @staticmethod
    # def _add_cols(df:pd.DataFrame, align_df:pd.DataFrame, cols:list=[], prefix:str='query'):
    #     assert df.index.is_unique
    #     for col in cols:
    #         if col not in df.columns:
    #             continue 
    #         align_df[f'{prefix}_{col}'] = align_df[f'{prefix}_id'].map(df[col])
    #     return align_df
    

    def align(self, query_df:pd.DataFrame, subject_df:pd.DataFrame=None, query_name:str=None, subject_name:str=None, output_dir:str='../data/', overwrite:bool=False, sensitivity:float=8, max_e_value:float=10, **kwargs):
        # MMSeqs align queryDB targetDB resultDB_pref resultDB_aln
        subject_name = query_name if (subject_name is None) else subject_name
        output_path = os.path.join(output_dir, f'{query_name}_{subject_name}_align.tsv')

        if not os.path.exists(output_path) or (overwrite):
            query_database_path, subject_database_path, prefilter_database_path = self._prefilter(query_df, subject_df=subject_df, query_name=query_name, subject_name=subject_name, sensitivity=sensitivity)
            
            output_database_name = f'{query_name}_{subject_name}_align_database'
            output_database_dir = self._make_database_dir(output_database_name)
            output_database_path = os.path.join(output_database_dir, output_database_name)
            
            print(f'MMSeqs.align: Running alignment on query database {os.path.basename(query_database_path)}.')
            cmd = f'mmseqs align {query_database_path} {subject_database_path} {prefilter_database_path} {output_database_path}'
            cmd += f' -e {max_e_value}'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(f'mmseqs convertalis {query_database_path} {subject_database_path} {output_database_path} {output_path}', shell=True, check=True, stdout=subprocess.DEVNULL)
    
        align_df = MMSeqs.load_align(output_path, **kwargs)
        # align_df = MMSeqs._add_cols(query_df, align_df, cols=add_cols, prefix='query')
        # align_df = MMSeqs._add_cols(subject_df, align_df, cols=add_cols, prefix='subject')
        return align_df
        

    def cluster(self, df:pd.DataFrame, name:str=None, output_dir:str='../data', sequence_identity:float=0.2, overwrite:bool=False):

        input_path = os.path.join(output_dir, name + '.faa')
        self.cleanup_files += [input_path]
        output_path = os.path.join(output_dir, name)

        if (not os.path.exists(output_path + '_cluster.tsv')) or overwrite:
            FASTAFile(df=df).write(input_path)
            subprocess.run(f'mmseqs easy-cluster {input_path} {output_path} {self.tmp_dir} --min-seq-id {sequence_identity}', shell=True, check=True, stdout=subprocess.DEVNULL)

        output_path = output_path + '_cluster.tsv'
        self.cleanup_files += [os.path.join(output_dir, file_name.format(name=name)) for file_name in MMSeqs.cleanup_files]

        return MMSeqs.load_cluster(output_path)

    @staticmethod
    def load_cluster(path:str, add_prefix:bool=False):
        df = pd.read_csv(path, delimiter='\t', names=MMSeqs.cluster_fields)
        cluster_ids = {rep:i for i, rep in enumerate(df.cluster_rep.unique())} # Add integer IDs for each cluster. 
        df['cluster_id'] = [cluster_ids[rep] for rep in df.cluster_rep]
        df = df.set_index('id')
        if add_prefix:
            df.columns = [f'{MMSeqs.prefix}_{col}' for col in df.columns]
        return df

    @staticmethod
    def load_align(path:str, add_prefix:bool=False, chunk_size:int=None):
        df = pd.read_csv(path, delimiter='\t', names=MMSeqs.align_fields, header=None, chunksize=chunk_size)
        if add_prefix:
            df.columns = [f'{MMSeqs.prefix}_{col}' for col in df.columns]
        return df
    
