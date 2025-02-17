import pandas as pd 
import os 
from src.files import FASTAFile
import subprocess
import shutil


class MMseqs():

    cleanup_files = ['{job_name}_rep_seq.fasta', '{job_name}_all_seqs.fasta']

    def __init__(self, tmp_dir:str='../data/tmp'):

        # Need a directory to store temporary files. If one does not already exist, create it in the working directory.
        self.tmp_dir = tmp_dir 
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
        self.cleanup_files = []

    def run(self, input_path:str, output_path:str, sequence_identity:float=0.2) -> str:

        job_name = os.path.basename(output_path)
        output_dir = os.path.dirname(output_path)
        self.cleanup_files += [os.path.join(output_dir, file_name.format(job_name=job_name)) for file_name in MMseqs.cleanup_files]

        cmd = f'mmseqs easy-cluster {input_path} {output_path} {self.tmp_dir} --min-seq-id {sequence_identity}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)

    def cleanup(self):
        for path in self.cleanup_files:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def cluster(self, df:pd.DataFrame, job_name:str=None, output_dir:str='../data/', sequence_identity:float=0.2, overwrite:bool=False, reps_only:bool=False):

        input_path = os.path.join(output_dir, job_name + '.faa')
        output_path = os.path.join(output_dir, job_name)

        if (not os.path.exists(output_path + '_cluster.tsv')) or overwrite:
            FASTAFile(df=df).write(input_path)
            self.cleanup_files.append(input_path)
            self.run(input_path, output_path, sequence_identity=sequence_identity)

        cluster_df = MMseqs.load(output_path + '_cluster.tsv', reps_only=reps_only)
        df = df.drop(columns=['cluster', 'cluster_rep'], errors='ignore')
        df = cluster_df.merge(df, how='left', left_index=True, right_index=True)
        return df

    @staticmethod
    def load(path:str, reps_only:bool=True):
        df = pd.read_csv(path, delimiter='\t', names=['cluster_rep', 'id'])
        cluster_ids = {rep:i for i, rep in enumerate(df.cluster_rep.unique())} # Add integer IDs for each cluster. 
        df['cluster'] = [cluster_ids[rep] for rep in df.cluster_rep]
        if reps_only:
            df = df.drop_duplicates('cluster_rep', keep='first')
        return df.set_index('id')
