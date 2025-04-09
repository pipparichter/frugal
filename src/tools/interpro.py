import pandas as pd 
from src import get_genome_id
import os 
from src.files import InterProScanFile, FASTAFile
import subprocess


class InterProScan():
    
    cmd = '/home/prichter/interproscan/interproscan-5.73-104.0/interproscan.sh'

    tmp_input_path = 'tmp_protein.faa'
    tmp_output_path = 'tmp_annotation.tsv'
    cleanup_files = [tmp_input_path, tmp_output_path]

    def __init__(self, tmp_dir='../data/interpro/tmp', print_cmd_only:bool=False):

        self.tmp_dir = tmp_dir
        self.cmd = InterProScan.cmd + f' --tempdir {tmp_dir} --formats tsv'
        self.cmd = self.cmd + ' --outfile {output_path} --input {input_path}'

        self.print_cmd_only = print_cmd_only

    def _run_append(self, input_path:str, output_path:str):

        df = FASTAFile(path=input_path).to_df()
        existing_ids = InterProScan._get_existing_ids(output_path)
        df = df[~df.index.isin(existing_ids)].copy() # Don't re-compute annotations. 
        print(f'InterProScan._run_append: {len(existing_ids)} already present in {output_path}. Computing annotations for {len(df)} new sequences.')

        # Write the remaining sequences to a FASTA file for the InterPro input.
        FASTAFile(df=df).write(InterProScan.tmp_input_path)
        
        cmd = self.cmd.format(input_path=input_path, output_path=InterProScan.tmp_output_path)
        if self.print_cmd_only:
            print(cmd)
            return
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        InterProScan._append(InterProScan.tmp_output_path, output_path)
    
    def run(self, input_path:str, name:str=None, output_dir:str='../data/interpro', overwrite:bool=False):

        output_file_name = f'{name}_annotation.tsv'
        output_path = os.path.join(output_dir, output_file_name)
        
        if os.path.exists(output_path) and (not overwrite):
            self._run_append(input_path, output_path)
        else:
            cmd = self.cmd.format(input_path=input_path, output_path=output_path)
            if self.print_cmd_only:
                print(cmd)
                return
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)

    @staticmethod
    def _append(tmp_output_path:str, output_path:str):
        with open(tmp_output_path, 'r') as f_tmp, open(output_path, 'a') as f:
            content = f_tmp.read()
            f.write('\n' + content)
    
    @staticmethod
    def _get_existing_ids(path:str) -> list:
        existing_ids = InterProScanFile(path).to_df().index.unique()
        return existing_ids

    def cleanup(self):
        for path in InterProScan.cleanup_files:
            if os.path.exists(path):
                os.remove(path)

        




