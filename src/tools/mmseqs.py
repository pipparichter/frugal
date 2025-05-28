import pandas as pd 
import os 
from src.files import FASTAFile
import subprocess
import shutil
import numpy as np 
from scipy.sparse import csr_matrix, save_npz, lil_matrix
import subprocess
from tqdm import tqdm

# TODO: Figure out what on Earth an LDDT score is. 


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


class MMSeqsBase():

    fields = dict()
    fields['query'] = 'query_id'
    fields['target'] = 'subject_id'
    fields['evalue'] = 'e_value'
    fields['fident'] = 'fraction_identical_matches'
    fields['pident'] = 'percent_identical_matches'
    fields['nident'] = 'n_identical_matches'
    fields['gapopen'] = 'n_gap_open_events'
    fields['alnlen'] = 'alignment_length'
    fields['qseq'] = 'query_seq'
    fields['tseq'] = 'subject_seq'
    fields['qaln'] = 'query_aligned_seq'
    fields['taln'] = 'subject_aligned_seq'
    fields['mismatch'] = 'n_mismatches'
    fields['raw'] = 'raw_alignment_score'
    fields['bits'] = 'bit_score'
    fields['qcov'] = 'query_coverage'
    fields['tcov'] = 'subject_coverage'

    minimal_field_codes = ['query', 'target', 'evalue', 'bits', 'fident', 'qseq', 'tseq', 'qcov', 'tcov', 'mismatch', 'alnlen']

    program = None 

    def __init__(self, tmp_dir:str='../data/tmp', database_dir:str='../data/databases/'):
        self.tmp_dir = tmp_dir
        self.database_dir = database_dir

    def _make_database(self, input_path:str, name:str=None):
        '''Create a database from a FASTA file or directory of PDB files.'''
        database_name = f'{name}_database_{self.program}'
        database_path = os.path.join(self.database_dir, database_name)
        if not os.path.exists(database_path):
            print(f'MMSeqsBase._make_database: Creating database {database_name} in {self.database_dir}')
            subprocess.run(f'{self.program} createdb {input_path} {database_path}', shell=True, check=True, stdout=subprocess.DEVNULL)
        return database_path 
    
    def _prefilter(self, query_database_path:str, subject_database_path, query_name:str=None, subject_name:str=None, sensitivity:float=None):

        prefilter_database_name = f'{query_name}_{subject_name}_prefilter_database_{self.program}'
        prefilter_database_path = os.path.join(self.database_dir, prefilter_database_name)

        cmd = f'{self.program} prefilter {query_database_path} {subject_database_path} {prefilter_database_path}'
        if sensitivity is not None:
            cmd += f' -s {sensitivity}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        return prefilter_database_path

    def _align(self, query_database_path:str, subject_database_path, query_name:str=None, subject_name:str=None, sensitivity:float=None, max_e_value:float=None):
        
        prefilter_database_path = self._prefilter(query_database_path, subject_database_path, query_name=query_name, subject_name=subject_name, sensitivity=sensitivity)
        output_database_name = f'{query_name}_{subject_name}_align_database_{self.program}'
        output_database_path = os.path.join(self.database_dir, output_database_name)

        cmd = f'{self.program} {'structurealign' if (self.program == 'foldseek') else 'align'} {query_database_path} {subject_database_path} {prefilter_database_path} {output_database_path} -a'
        if max_e_value is not None:
            cmd += f' -e {max_e_value}'

        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        return output_database_path
        
    def _make_tsv_output(self, output_path:str, output_database_path:str=None, query_database_path:str=None, subject_database_path:str=None):
        fmt = f'--format-output "{','.join(self.minimal_field_codes)}"' # Use a custom output format. 
        cmd = f'{self.program} convertalis {query_database_path} {subject_database_path} {output_database_path} {output_path}'
        cmd += f' {fmt}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)

    @classmethod
    def load(cls, path:str, chunk_size:int=None):
        names = [cls.fields[field_code] for field_code in cls.minimal_field_codes]
        df = pd.read_csv(path, delimiter='\t', names=names, header=None, chunksize=chunk_size)
        return df


class MMSeqs(MMSeqsBase):

    program = 'mmseqs'

    def _make_fasta(self, df:pd.DataFrame, name:str=None):
        path = os.path.join(self.database_dir, f'{name}.faa')
        FASTAFile(df=df).write(path)
        return path

    def align(self, query_df:pd.DataFrame, subject_df:pd.DataFrame=None, query_name:str=None, subject_name:str=None, output_dir:str='.', overwrite:bool=False, sensitivity:float=8, max_e_value:float=10, **kwargs):
        # MMSeqs align queryDB targetDB resultDB_pref resultDB_aln
        subject_name = query_name if (subject_name is None) else subject_name
        subject_df = query_df if (subject_df is None) else subject_df

        output_path = os.path.join(output_dir, f'{query_name}_{subject_name}_align_{self.program}.tsv')

        query_path, subject_path= self._make_fasta(query_df, name=query_name), self._make_fasta(subject_df, name=subject_name)
        query_database_path = self._make_database(query_path, name=query_name)
        subject_database_path = self._make_database(subject_path, name=subject_name)

        if not os.path.exists(output_path) or overwrite:
            output_database_path = self._align(query_database_path, subject_database_path, query_name=query_name, subject_name=subject_name, sensitivity=sensitivity, max_e_value=max_e_value)
            self._make_tsv_output(output_path, output_database_path=output_database_path, query_database_path=query_database_path, subject_database_path=subject_database_path)
            
        align_df = MMSeqsBase.load(output_path, **kwargs)
        return align_df
        
    def cluster(self, df:pd.DataFrame, name:str=None, sequence_identity:float=0.5, overwrite:bool=False, output_dir:str='.'):

        input_path = self._make_fasta(df, name=name)
        output_path = os.path.join(output_dir, name)

        if (not os.path.exists(output_path + '_cluster.tsv')) or overwrite:
            subprocess.run(f'mmseqs easy-cluster {input_path} {output_path} {self.tmp_dir} --min-seq-id {sequence_identity}', shell=True, check=True, stdout=subprocess.DEVNULL)

        output_path = output_path + '_cluster.tsv'
        return MMSeqs.load_cluster(output_path)

    @staticmethod
    def load_cluster(path:str):
        df = pd.read_csv(path, delimiter='\t', names=['cluster_rep_id', 'id'])
        cluster_ids = {id_:i for i, id_ in enumerate(df.cluster_rep_id.unique())} # Add integer IDs for each cluster. 
        df['cluster_id'] = [cluster_ids[id_] for id_ in df.cluster_rep_id]
        df = df.set_index('id')
        return df
    


class Foldseek(MMSeqsBase):

    fields = MMSeqsBase.fields
    fields['qca'] = 'query_calpha_coordinates'
    fields['tca'] = 'subject_calpha_coordinates'
    fields['alntmscore'] = 'tm_score' # This is the TM-score of the alignment. 
    fields['qtmscore'] = 'query_normalized_tm_score'
    fields['ttmscore'] = 'subject_normalized_tm_score'
    fields['u'] = 'rotation_matrix'
    fields['t'] = 'translation_vector'
    fields['lddt'] = 'average_lddt'
    fields['lddtfull'] = 'per_position_lddt'
    fields['prob'] = 'probability_homologous'

    minimal_field_codes = MMSeqsBase.minimal_field_codes + ['prob', 'lddt', 'alntmscore']

    program = 'foldseek'

    def __init__(self, tmp_dir:str='../data/tmp', database_dir:str='../data/databases/'):
        super().__init__(tmp_dir=tmp_dir, database_dir=database_dir)


    def align(self, query_path:pd.DataFrame, subject_path:pd.DataFrame=None, query_name:str=None, subject_name:str=None, output_dir:str='.', overwrite:bool=False, sensitivity:float=8, max_e_value:float=10, **kwargs):
        # MMSeqs align queryDB targetDB resultDB_pref resultDB_aln
        subject_name = query_name if (subject_name is None) else subject_name
        output_path = os.path.join(output_dir, f'{query_name}_{subject_name}_align_{self.program}.tsv')

        query_database_path = self._make_database(query_path, name=query_name)
        subject_database_path = self._make_database(subject_path, name=subject_name)

        if not os.path.exists(output_path) or overwrite:
            output_database_path = self._align(query_database_path, subject_database_path, query_name=query_name, subject_name=subject_name, sensitivity=sensitivity, max_e_value=max_e_value)
            self._make_tsv_output(output_path, output_database_path=output_database_path, query_database_path=query_database_path, subject_database_path=subject_database_path)
            
        align_df = Foldseek.load(output_path)
        return align_df