import os 
import pandas as pd 
import subprocess
from tqdm import tqdm 
import shutil
from typing import List
import numpy as np 
import warnings
import glob
from src.files import GBFFFile
from src import get_genome_id


class NCBIDatasets():

    cleanup_files = ['README.md', 'md5sum.txt', 'ncbi.zip']
    cleanup_dirs = ['ncbi_dataset']

    include = ['gbff', 'genome', 'protein']

    src_dir = 'ncbi_dataset/data'
    src_file_names = {'gbff':'genomic.gbff', 'genome':'*genomic.fna', 'protein':'*protein.faa'}
    dst_file_names = {'gbff':'{genome_id}_genomic.gbff', 'genome':'{genome_id}_genomic.fna', 'protein':'{genome_id}_protein.faa'}

    def __init__(self, genome_dir:str='../data/genomes', gbff_dir='../data/ncbi', protein_dir:str=None):

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

    @staticmethod
    def make_database(path:str, dir_:str='../data/ncbi', feature:str='CDS'):

        paths = tqdm(glob.glob(os.path.join(dir_, '*')), desc=f'NCBIDatasets.make_database: Reading GBFF files in {dir_}')
        df = pd.concat([GBFFFile(path).to_df().assign(genome_id=get_genome_id(path)) for path in paths])

        if (feature is not None): # Filter for a specific feature, if specified.
            df = df[df.feature == feature]

        print(f'NCBIDatasets.make_database: Writing NCBI data to {path}')
        df.to_csv(path)
        

def fix_b_subtilis(database_path:str='../data/ncbi_database_cds.csv'):
    genome_id = 'GCF_000009045.1' 

    df = pd.read_csv(database_path, index_col=0, dtype={'partial':str}, low_memory=False)
    bsub_df = df[df.genome_id == genome_id].copy()
    df = df[df.genome_id != genome_id].copy()

    evidence_types = []
    for row in bsub_df.itertuples():
        if ('Evidence 1' in row.note) or ('Evidence 2' in row.note):
            evidence_types.append('experiment')
        elif ('Evidence 4' in row.note) or ('Evidence 3' in row.note):
            evidence_types.append('similar to sequence')
        else:
            evidence_types.append('ab initio prediction')

    bsub_df['evidence_type'] = evidence_types
    df = pd.concat([df, bsub_df])
    print(f'fix_b_subtilis: Writing modified database to {database_path}')
    df.to_csv(database_path)



