import pandas as pd 
from src import get_genome_id
import os 
from src.files import InterProScanFile, FASTAFile
import subprocess

def remove_asterisks(*paths):
    '''Prodigal adds an asterisk marking the terminal end of each amino acid sequence by default. These are not compatible 
    with tools like InterProScan, so should be removed.'''
    for path in paths:
        with open(path, 'r') as f:
            content = f.read()
        content = content.replace('*', '')
        with open(path, 'w') as f:
            f.write(content)


class InterProScan():
    
    # cmd = '/home/prichter/interproscan/interproscan-5.73-104.0/interproscan.sh'
    cmd = '/home/prichter/interproscan/interproscan-5.74-105.0/interproscan.sh'

    tmp_input_path = 'tmp_protein.faa'
    tmp_output_path = 'tmp_annotation.tsv'
    cleanup_files = [tmp_input_path, tmp_output_path]

    def __init__(self, tmp_dir='../data/interpro/tmp', print_cmd_only:bool=False):

        self.tmp_dir = tmp_dir
        self.cmd = InterProScan.cmd + f' --tempdir {tmp_dir} --formats tsv'
        self.cmd = self.cmd + ' --outfile {output_path} --input {input_path}'

        self.print_cmd_only = print_cmd_only

    # def _run_append(self, input_path:str, output_path:str):

    #     df = FASTAFile(path=input_path).to_df()
    #     existing_ids = InterProScan._get_existing_ids(output_path)
    #     df = df[~df.index.isin(existing_ids)].copy() # Don't re-compute annotations. 
    #     print(f'InterProScan._run_append: {len(existing_ids)} already present in {output_path}. Computing annotations for {len(df)} new sequences.')

    #     # Write the remaining sequences to a FASTA file for the InterPro input.
    #     FASTAFile(df=df).write(InterProScan.tmp_input_path)
        
    #     cmd = self.cmd.format(input_path=input_path, output_path=InterProScan.tmp_output_path)
    #     if self.print_cmd_only:
    #         print(cmd)
    #         return
    #     subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
    #     InterProScan._append(InterProScan.tmp_output_path, output_path)
    
    def run(self, input_path:str, output_path:str):
        remove_asterisks(input_path) # Make sure there are no leftover asterisks from Prodigal. 

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

        




