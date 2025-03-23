import os 
import pandas as pd 
import subprocess
from tqdm import tqdm 
import shutil
from typing import List
import numpy as np 
import warnings
import glob
from src.files import GBFFFile, FASTAFile
from src import get_genome_id
import io 
import time 
from src import fillna

# datasets summary genome taxon 2 --reference --annotated --assembly-level complete --mag exclude --assembly-source RefSeq --exclude-atypical --report sequence --as-json-lines | dataformat tsv genome-seq --fields accession,genbank-seq-acc,refseq-seq-acc,chr-name,seq-length,gc-percent

class NCBIDatasets():
    taxonomy_fields = ['Taxid', 'Tax name', 'Authority', 'Rank', 'Basionym', 'Basionym authority', 'Curator common name', 'Has type material', 'Group name', 'Superkingdom name', 'Superkingdom taxid', 'Kingdom name', 'Kingdom taxid', 'Phylum name', 'Phylum taxid', 'Class name', 'Class taxid', 'Order name', 'Order taxid', 'Family name', 'Family taxid', 'Genus name', 'Genus taxid', 'Species name', 'Species taxid'] 
    cleanup_files = ['README.md', 'md5sum.txt', 'ncbi.zip']
    cleanup_dirs = ['ncbi_dataset']

    include = ['gbff', 'genome'] # , 'protein']

    src_dir = 'ncbi_dataset/data'
    src_file_names = {'gbff':'genomic.gbff', 'genome':'*genomic.fna', 'protein':'*protein.faa', 'gene':'gene.fna'}
    dst_file_names = {'gbff':'{genome_id}_genomic.gbff', 'genome':'{genome_id}_genomic.fna', 'protein':'{genome_id}_protein.faa'}

    def __init__(self):
        pass

    @staticmethod
    def _get_metadata(ids:list, cmd:str=None, path:str=None, chunk_size:int=20) -> pd.DataFrame:

        df = list()
        if os.path.exists(path):
            df_ = pd.read_csv(path, sep='\t')
            print(f'NCBIDatasets._get_metadata: Found metadata entries for {len(df_)} IDs already in {path}.')
            ids = [id_ for id_ in ids if id_ not in df_.iloc[:, 0].values] # Don't repeatedly download the same ID.
            df.append(df_)

        n_chunks = 0 if (len(ids) == 0) else len((ids) // chunk_size + 1) # Handle case where ID list is empty.
        ids = [str(id_) for id_ in ids] # Convert the IDs to strings for joining, needed for the taxonomy IDs. 
        ids = [','.join(ids[i:i + chunk_size]) for i in range(0, n_chunks * chunk_size, chunk_size)]

        for id_ in tqdm(ids, desc='NCBIDatasets._get_metadata: Downloading metadata.'):
            try:
                output = subprocess.run(cmd.format(id_=id_), shell=True, check=True, capture_output=True)
                content = output.stdout.decode('utf-8').strip().split('\n')
                df_ = pd.read_csv(io.StringIO('\n'.join(content)), sep='\t')
                df_ = df_.drop(columns=['Query'], errors='ignore') # Drop the Query column, which is redundant. Only present when getting taxonomy metadata. 
                df.append(df_)
            except pd.errors.EmptyDataError as err: # Raised when the call to pd.read_csv fails, I think due to nothing written to stdout.
                print(f'NCBIDatasets._get_metadata: Failed on query {id_}. NCBI returned the following error message.')
                print(output.stderr.decode('utf-8'))
                return pd.concat(df) # Return everything that has been downloaded already to not lose progress.

        return pd.concat(df)

    def _get_taxonomy_metadata(self, taxonomy_ids:list, path:str='../data/ncbi_taxonomy_metadata.tsv'):
        
        cmd = 'datasets summary taxonomy taxon {id_} --as-json-lines | dataformat tsv taxonomy --template tax-summary'
        df = NCBIDatasets._get_metadata(taxonomy_ids, cmd=cmd, path=path)
        df = df.set_index('Taxid') 
        df.to_csv(path, sep='\t')
    
    def _get_genome_metadata(self, genome_ids:list, path:str='../data/ncbi_genome_metadata.tsv'):

        # https://www.ncbi.nlm.nih.gov/datasets/docs/v2/command-line-tools/using-dataformat/genome-data-reports/ 
        fields = 'accession,checkm-completeness,annotinfo-method,annotinfo-pipeline,assmstats-gc-percent,assmstats-total-sequence-len,'
        fields += 'assmstats-number-of-contigs,organism-tax-id,annotinfo-featcount-gene-pseudogene,annotinfo-featcount-gene-protein-coding,'
        fields += 'annotinfo-featcount-gene-non-coding'

        cmd = 'datasets summary genome accession {id_} --report genome --as-json-lines | dataformat tsv genome --fields ' + fields
        df = NCBIDatasets._get_metadata(genome_ids, cmd=cmd, path=path)
        df = fillna(df, rules={str:'none'}, check=False)
        df = df.set_index('Assembly Accession') 
        df.to_csv(path, sep='\t')
    
    def _get_genome(self, genome_ids:list, include:list=['gbff', 'genome'], dirs={'genome':'../data/ncbi/genomes', 'gbff':'../data/ncbi/gbffs'}):

        pbar = tqdm(genome_ids)
        for genome_id in pbar:
            pbar.set_description(f'NCBIDatasets._get_genome: Downloading data for {genome_id}.')
            src_paths = [os.path.join(NCBIDatasets.src_dir, genome_id, NCBIDatasets.src_file_names[i]) for i in include]
            dst_paths = [os.path.join(dirs[i], NCBIDatasets.dst_file_names[i].format(genome_id=genome_id)) for i in include]

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
                print(f'NCBIDatasets._get_genome: Failed to download data for {genome_id}. Returned error message "{err}"')

    def _get_proteins(self, protein_ids:list, include=['protein'], path:str=None, chunk_size:int=20):
        # Need to break into chunks because the API doesn't support more than a handful of sequences. 
        n_chunks = np.ceil(len(protein_ids) / chunk_size).astype(int)
        protein_ids = [','.join(protein_ids[i:i + chunk_size]) for i in range(0, n_chunks * chunk_size, chunk_size)]
        
        df = list()
        src_path = os.path.join(NCBIDatasets.src_dir, 'protein.faa')
        for protein_ids_ in tqdm(protein_ids, 'NCBIDatasets._get_proteins'):
            cmd = f"datasets download gene accession {protein_ids_} --include protein --filename ncbi.zip"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # The -o option means that the ncbi.zip directory from the previous pass will be overwritten without prompting. 
            subprocess.run(f'unzip -o ncbi.zip -d .', shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            df.append(FASTAFile(path=src_path).to_df(prodigal_output=False))
        df = pd.concat(df)
        
        if path is not None:
            print(f'NCBIDatasets._get_proteins: Proteins saved to {path}')
            FASTAFile(df=df).write(path)
        return df 

    def run(self, genome_ids:list=None, taxonomy_ids:list=None, protein_ids:list=None, metadata:bool=False, **kwargs):

        if not (genome_ids is None) and (not metadata):
            return self._get_genome(genome_ids, **kwargs)
        if not (genome_ids is None):
            return self._get_genome_metadata(list(genome_ids), **kwargs)
        if not (taxonomy_ids is None):
            return self._get_taxonomy_metadata(taxonomy_ids, **kwargs)
        if not (protein_ids is None):
            return self._get_proteins(list(protein_ids), **kwargs)

    def cleanup(self):
        for file in NCBIDatasets.cleanup_files:
            if os.path.exists(file):
                os.remove(file)
        for dir_ in NCBIDatasets.cleanup_dirs:
            if os.path.isdir(dir_):
                shutil.rmtree(dir_)


        

def fix_b_subtilis(database_path:str='../data/ncbi_cds.csv'):
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


    # @staticmethod
    # def make_database(path:str, dir_:str='../data/ncbi', feature:str='CDS'):

    #     paths = tqdm(glob.glob(os.path.join(dir_, '*')), desc=f'NCBIDatasets.make_database: Reading GBFF files in {dir_}')
    #     df = pd.concat([GBFFFile(path).to_df().assign(genome_id=get_genome_id(path)) for path in paths])

    #     if (feature is not None): # Filter for a specific feature, if specified.
    #         df = df[df.feature == feature]

    #     print(f'NCBIDatasets.make_database: Writing NCBI data to {path}')
    #     df.to_csv(path)
