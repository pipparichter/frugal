import os 
import pandas as pd 
import subprocess
from tqdm import tqdm 
import shutil
import numpy as np 
from src.files import FASTAFile
import io 
from src import fillna
import requests 
import json
import re 

class NCBI():
    taxonomy_fields = ['Taxid', 'Tax name', 'Authority', 'Rank', 'Basionym', 'Basionym authority', 'Curator common name', 'Has type material', 'Group name', 'Superkingdom name', 'Superkingdom taxid', 'Kingdom name', 'Kingdom taxid', 'Phylum name', 'Phylum taxid', 'Class name', 'Class taxid', 'Order name', 'Order taxid', 'Family name', 'Family taxid', 'Genus name', 'Genus taxid', 'Species name', 'Species taxid'] 
    cleanup_files = ['README.md', 'md5sum.txt', 'ncbi.zip']
    cleanup_dirs = ['ncbi_dataset']

    src_dir = 'ncbi_dataset/data'
    src_file_names = {'gbff':'genomic.gbff', 'genome':'*genomic.fna', 'protein':'*protein.faa', 'gene':'gene.fna'}
    dst_file_names = {'gbff':'{genome_id}_genomic.gbff', 'genome':'{genome_id}_genomic.fna', 'protein':'{genome_id}_protein.faa'}

    def __init__(self):
        pass

    @staticmethod
    def _get_metadata(ids:list, cmd:str=None, path:str=None, chunk_size:int=20) -> pd.DataFrame:

        df = list()
        if (path is not None) and os.path.exists(path):
            df_ = pd.read_csv(path, sep='\t')
            ids = [id_ for id_ in ids if (id_ not in df_.iloc[:, 0].values)] # Don't repeatedly download the same ID.
            print(f'NCBI._get_metadata: Found metadata entries for {len(df_)} IDs already in {path}. Downloading metadata for {len(ids)} entries.')
            df.append(df_)

        n_chunks = 0 if (len(ids) == 0) else len(ids) // chunk_size + 1 # Handle case where ID list is empty.
        ids = [str(id_) for id_ in ids] # Convert the IDs to strings for joining, needed for the taxonomy IDs. 
        ids = [','.join(ids[i:i + chunk_size]) for i in range(0, n_chunks * chunk_size, chunk_size)]

        for id_ in tqdm(ids, desc='NCBI._get_metadata: Downloading metadata.'):
            try:
                output = subprocess.run(cmd.format(id_=id_), shell=True, check=True, capture_output=True)
                content = output.stdout.decode('utf-8').strip().split('\n')
                df_ = pd.read_csv(io.StringIO('\n'.join(content)), sep='\t')
                df_ = df_.drop(columns=['Query'], errors='ignore') # Drop the Query column, which is redundant. Only present when getting taxonomy metadata. 
                df.append(df_)
            except pd.errors.EmptyDataError as err: # Raised when the call to pd.read_csv fails, I think due to nothing written to stdout.
                print(f'NCBI._get_metadata: Failed on query {id_}. NCBI returned the following error message.')
                print(output.stderr.decode('utf-8'))
                return pd.concat(df) # Return everything that has been downloaded already to not lose progress.
        return pd.concat(df)

    def get_taxonomy_metadata(self, taxonomy_ids:list, path:str=None):

        cmd = 'datasets summary taxonomy taxon {id_} --as-json-lines | dataformat tsv taxonomy --template tax-summary'
        df = NCBI._get_metadata(taxonomy_ids, cmd=cmd, path=path)
        df = df.set_index('Taxid')
        df = df[~df.index.duplicated(keep='first')].copy()

        if path is not None:
            print(f'NCBI.get_taxonomy_metadata: Writing metadata for {len(df)} taxa to {path}')
            df.to_csv(path, sep='\t')
        return df
    
    def get_genome_metadata(self, genome_ids:list, path:str='../data/ncbi_genome_metadata.tsv'):

        # https://www.ncbi.nlm.nih.gov/datasets/docs/v2/command-line-tools/using-dataformat/genome-data-reports/ 
        fields = 'accession,checkm-completeness,annotinfo-method,annotinfo-pipeline,assmstats-gc-percent,assmstats-total-sequence-len,'
        fields += 'assmstats-number-of-contigs,organism-tax-id,annotinfo-featcount-gene-pseudogene,annotinfo-featcount-gene-protein-coding,'
        fields += 'annotinfo-featcount-gene-non-coding'

        cmd = 'datasets summary genome accession {id_} --report genome --as-json-lines | dataformat tsv genome --fields ' + fields
        df = NCBI._get_metadata(genome_ids, cmd=cmd, path=path)
        df = fillna(df, rules={str:'none'}, check=False)
        df = df.set_index('Assembly Accession') 
        df.to_csv(path, sep='\t')
    
    def get_genomes(self, genome_ids:list, include:list=['gbff', 'genome'], dirs={'genome':'../data/ncbi/genomes', 'gbff':'../data/ncbi/gbffs'}):

        pbar = tqdm(genome_ids)
        for genome_id in pbar:
            pbar.set_description(f'NCBI.get_genomes: Downloading data for {genome_id}.')
            src_paths = [os.path.join(NCBI.src_dir, genome_id, NCBI.src_file_names[i]) for i in include]
            dst_paths = [os.path.join(dirs[i], NCBI.dst_file_names[i].format(genome_id=genome_id)) for i in include]

            if np.all([os.path.exists(path) for path in dst_paths]): # Skip if already downloaded. 
                continue

            cmd = f"datasets download genome accession {genome_id} --filename ncbi.zip --include {','.join(include)} --no-progressbar"
            try:
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
                # The -o option means that the ncbi.zip directory from the previous pass will be overwritten without prompting. 
                subprocess.run(f'unzip -o ncbi.zip -d .', shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Unpack the downloaded NCBI data package. 
                for src_path, dst_path in zip(src_paths, dst_paths):
                    subprocess.run(f'cp {src_path} {dst_path}', shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as err:
                print(f'NCBI.get_genomes: Failed to download data for {genome_id}. Returned error message "{err}"')

    def get_nuccore_entries(self, nuccore_ids:list, chunk_size:int=50, dir_:str='../data/results/rpoz/gbffs'):
        pbar = tqdm(total=len(nuccore_ids), desc='get_nuccore')
        nuccore_ids = np.array_split(nuccore_ids, len(nuccore_ids) // chunk_size + 1)

        for ids in nuccore_ids:
            url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={ids}&rettype=gbwithparts&retmode=text'.format(ids=','.join(ids))
            content = requests.get(url).text
            for id_, content_ in zip(ids, re.split('(?=LOCUS)', content)[1:]): # Split by LOCUS so that each is written individually to a file. 
                with open(os.path.join(dir_, f'{id_}_genomic.gbff'), 'w') as f:
                    f.write(content_)
            pbar.update(len(ids))

    def cleanup(self):
        for file in NCBI.cleanup_files:
            if os.path.exists(file):
                os.remove(file)
        for dir_ in NCBI.cleanup_dirs:
            if os.path.isdir(dir_):
                shutil.rmtree(dir_)