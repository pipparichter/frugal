import os 
import pandas as pd 
import subprocess
from tqdm import tqdm 
import shutil
from typing import List
import numpy as np 


class NCBIDatasets():

    cleanup_files = ['README.md', 'md5sum.txt', 'ncbi.zip']
    cleanup_dirs = ['ncbi_dataset']

    include = ['gbff', 'genome', 'protein']

    src_dir = 'ncbi_dataset/data'
    src_file_names = {'gbff':'genomic.gbff', 'genome':'*genomic.fna', 'protein':'*protein.faa'}
    dst_file_names = {'gbff':'{genome_id}_genomic.gbff', 'genome':'{genome_id}_genomic.fna', 'protein':'{genome_id}_protein.faa'}

    def __init__(self, genome_dir:str='../data/genomes', gbff_dir='../data/refseq', protein_dir:str=None):

        self.dst_dirs = dict()
        self.dst_dirs['gbff'] = gbff_dir 
        self.dst_dirs['genome'] = genome_dir
        self.dst_dirs['protein'] = protein_dir

    def run(self, genome_ids:List[str], include:List[str]=['gbff', 'genome']):
        
        pbar = tqdm(genome_ids, desc='NCBIDatasets.run: Downloading data from NCBI.')
        for genome_id in pbar:
            src_paths = [os.path.join(NCBIDatasets.src_dir, genome_id, NCBIDatasets.src_file_names[i]) for i in include]
            dst_paths = [os.path.join(self.dst_dirs[i], NCBIDatasets.dst_file_names[i].format(genome_id=genome_id)) for i in include]

            if np.all([os.path.exists(path) for path in dst_paths]): # Skip if already downloaded. 
                continue

            cmd = f"datasets download genome accession {genome_id} --filename ncbi.zip --include {','.join(include)} --no-progressbar"
            pbar.set_description(f'NCBIDatasets.run: Downloading data for {genome_id}.')
            try:
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
                # The -o option means that the ncbi.zip directory from the previous pass will be overwritten without prompting. 
                subprocess.run(f'unzip -o ncbi.zip -d .', shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                for src_path, dst_path in zip(src_paths, dst_paths):
                    subprocess.run(f'cp {src_path} {dst_path}', shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                print(f'NCBIDatasets.run: Failed to download data for {genome_id}.')

    def cleanup(self):
        for file in NCBIDatasets.cleanup_files:
            if os.path.exists(file):
                os.remove(file)
        for dir_ in NCBIDatasets.cleanup_dirs:
            if os.path.isdir(dir_):
                shutil.rmtree(dir_)



