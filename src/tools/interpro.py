import pandas as pd 
from src import get_genome_id
import os 
from src.files import InterProScanFile, FASTAFile
import subprocess


class InterProScan():
    
    cmd = '/home/prichter/interproscan/interproscan-5.73-104.0/interproscan.sh'
    cleanup_files = ['tmp.faa', 'tmp_annotation.tsv']

    def __init__(self, tmp_dir='../data/interpro/tmp'):

        self.tmp_dir = tmp_dir
        self.cmd = InterProScan.cmd + f' --tempdir {tmp_dir} --formats tsv'
        self.cmd = self.cmd + ' --outfile {output_path} --input {input_path}' 
    
    # def run(self, df:pd.DataFrame, name:str=None, output_dir:str='../data/interpro', overwrite:bool=False):
    def run(self, input_path:pd.DataFrame, name:str=None, output_dir:str='../data/interpro', overwrite:bool=False):

        name = get_genome_id(input_path) if (name is None) else name 

        output_file_name = f'{name}_annotation.tsv'
        output_path = os.path.join(output_dir, output_file_name)
        
        if os.path.exists(output_path):
            df = FASTAFile(path=input_path).to_df()
            existing_ids = InterProScanFile(output_path).to_df().index.unique()
            df = df[~df.index.isin(existing_ids)].copy() # Don't re-compute annotations. 
            print(f'InterProScan.run: {len(existing_ids)} already present in {output_path}. Computing annotations for {len(df)} new sequences.')
            # Write the remaining sequences to a FASTA file for the InterPro input.
            input_path = 'tmp.faa'
            FASTAFile(df=df).write(input_path)
        
        tmp_output_path = 'tmp_annotation.tsv'

        cmd = self.cmd.format(input_path=input_path, output_path=tmp_output_path)
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)

        InterProScan._append(tmp_output_path, output_path)

    @staticmethod
    def _append(tmp_output_path:str, output_path:str):
        with open(tmp_output_path, 'r') as f_tmp, open(output_path, 'a') as f:
            content = f_tmp.read()
            f.write('\n' + content)

    def cleanup(self):
        pass 

        




