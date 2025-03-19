import pandas as pd 
import os 
from src.files import FASTAFile
import subprocess
import shutil


class MMSeqs():

    cleanup_files = ['{job_name}_rep_seq.fasta', '{job_name}_all_seqs.fasta']

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

    def run(self, df:pd.DataFrame, job_name:str=None, output_dir:str=None, module:str='cluster', **kwargs) -> str:

        funcs = {'cluster':self.cluster, 'align':self.align}
        load_funcs = {'cluster':MMSeqs.load_cluster, 'align':MMSeqs.load_align}

        output_path = funcs[module](df, job_name=job_name, output_dir=output_dir, **kwargs)
        output_df = load_funcs[module](output_path, **kwargs)
        
        self.cleanup_files += [os.path.join(output_dir, file_name.format(job_name=job_name)) for file_name in MMSeqs.cleanup_files]

        return output_df

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

    def _make_database(self, df:pd.DataFrame, job_name:str=None, overwrite:bool=False):
        '''Create an mmseqs database from a FASTA file, using the sequences in the input DataFrame.'''
        database_name = f'{job_name}_database'
        database_dir = self._make_database_dir(database_name)
        database_path = os.path.join(database_dir, database_name)

        input_path = os.path.join(self.dir_, job_name + '.faa')

        if (not os.path.exists(database_path)) or overwrite:
            print(f'MMSeqs._make_database: Creating database {database_name} in {database_dir}')
            FASTAFile(df=df).write(input_path)
            subprocess.run(f'mmseqs createdb {input_path} {database_path}', shell=True, check=True, stdout=subprocess.DEVNULL)

        return database_path 

    def _prefilter(self, df:pd.DataFrame, job_name:str=None, overwrite:bool=False, sensitivity:float=None):

        output_database_name = f'{job_name}_prefilter_database'
        output_database_dir = self._make_database_dir(output_database_name)
        output_database_path = os.path.join(output_database_dir, output_database_name)

        input_database_path = self._make_database(df, job_name=job_name)

        if (not os.path.exists(output_database_path)) or overwrite:
            cmd = f'mmseqs prefilter {input_database_path} {input_database_path} {output_database_path}'
            if sensitivity is not None:
                cmd += f' -s {sensitivity}'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        
        return output_database_path


    def align(self, df:pd.DataFrame, job_name:str=None, output_dir:str='../data/', overwrite:bool=False, sensitivity:float=None, max_e_value:float=1e-3, **kwargs):
        # MMSeqs align queryDB targetDB resultDB_pref resultDB_aln
        
        input_database_path = self._make_database(df, job_name=job_name)
        prefilter_database_path = self._prefilter(df, job_name=job_name, sensitivity=sensitivity)
        
        output_database_name = f'{job_name}_align_database'
        output_database_dir = self._make_database_dir(output_database_name)
        output_database_path = os.path.join(output_database_dir, output_database_name)
        
        if (not os.path.exists(output_database_path)) or overwrite:
            print(f'MMSeqs.align: Running alignment on query database {os.path.basename(input_database_path)}.')
            cmd = f'mmseqs align {input_database_path} {input_database_path} {prefilter_database_path} {output_database_path}'
            cmd += f' -e {max_e_value}'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        
        # Convert the MMSeqs database output to a TSV file. 
        output_path = os.path.join(output_dir, f'{job_name}_align.tsv')
        subprocess.run(f'mmseqs convertalis {input_database_path} {input_database_path} {output_database_path} {output_path}', shell=True, check=True, stdout=subprocess.DEVNULL)
        return output_path

    def cluster(self, df:pd.DataFrame, job_name:str=None, output_dir:str=None, sequence_identity:float=0.2, overwrite:bool=False, **kwargs):

        input_path = os.path.join(output_dir, job_name + '.faa')
        self.cleanup_files += [input_path]
        output_path = os.path.join(output_dir, job_name)

        if (not os.path.exists(output_path + '_cluster.tsv')) or overwrite:
            FASTAFile(df=df).write(input_path)
            subprocess.run(f'mmseqs easy-cluster {input_path} {output_path} {self.tmp_dir} --min-seq-id {sequence_identity}', shell=True, check=True, stdout=subprocess.DEVNULL)

        return output_path + '_cluster.tsv'

    @staticmethod
    def load_cluster(path:str, add_prefix:bool=True, **kwargs):
        df = pd.read_csv(path, delimiter='\t', names=MMSeqs.cluster_fields)
        cluster_ids = {rep:i for i, rep in enumerate(df.cluster_rep.unique())} # Add integer IDs for each cluster. 
        df['cluster_label'] = [cluster_ids[rep] for rep in df.cluster_rep]
        df = df.set_index('id')
        if add_prefix:
            df.columns = [f'{MMSeqs.prefix}_{col}' for col in df.columns]
        return df

    @staticmethod
    def load_align(path:str, add_prefix:bool=False, **kwargs):
        df = pd.read_csv(path, delimiter='\t', names=MMSeqs.align_fields, header=None)
        df = df.set_index('query_id')

        if add_prefix:
            df.columns = [f'{MMSeqs.prefix}_{col}' for col in df.columns]
        return df