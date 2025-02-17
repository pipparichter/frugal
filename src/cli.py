import numpy as np 
import pandas as pd 
from src.build import Build 
from setuptools import Command
import argparse
from src.genome import ReferenceGenome
from src.files import FASTAFile
from src.embedders import get_embedder
import re
import glob
from tqdm import tqdm 
import os
from src import get_genome_id

        

def build():
    pass 


def ref():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', nargs='+', type=str)
    parser.add_argument('--output-dir', '-o', default='../data/ref.out/', type=str)
    parser.add_argument('--ref-dir', default='../data/refseq/', type=str)
    parser.add_argument('--prodigal-output', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # pbar = tqdm(glob.glob(args.input-path), desc='ref: Searching reference genomes.')
    pbar = tqdm(args.input_path, desc='ref: Searching reference genomes.')

    for path in pbar:

        genome_id = get_genome_id(path)
        assert genome_id is not None, f'ref: Could not extract a genome ID from the input path {path}.'

        output_path = os.path.join(args.output_dir, f'{genome_id}.ref.csv')
        pbar.set_description(f'ref: Searching reference genome for {genome_id}.')
        
        if os.path.exists(output_path) and (not args.overwrite):
            continue

        ref_path = os.path.join(args.ref_dir, f'{genome_id}_genomic.gbff')
        genome = ReferenceGenome(ref_path, genome_id=genome_id)

        df = FASTAFile(path).to_df(prodigal_output=args.prodigal_output)
        results_df = genome.search(df, verbose=False)

        results_df.to_csv(output_path)

    print(f'ref: Search complete. Output written to {args.output_dir}')


# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 300GB --partition gpu --gres gpu:1 --time 24:00:00 --wrap "embed -i ../data/filter_dataset_train.csv"
def embed():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', type=str)
    parser.add_argument('--output-path', '-o', default=None, type=str)
    parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    output_path = args.output_path if (args.output_path is not None) else args.input_path.replace('.csv', '.h5')

    # Sort the sequences in order of length, so that the longest sequences are first. This is so that the 
    # embedder works more efficiently, and if it fails, it fails early in the embedding process.
    df = pd.read_csv(args.input_path, index_col=0)
    df = df.iloc[np.argsort(df.seq.apply(len))[::-1]]

    store = pd.HDFStore(output_path, mode='a' if (not args.overwrite) else 'w', table=True) # Should confirm that the file already exists. 
    existing_keys = [key.replace('/', '') for key in store.keys()]

    if 'metadata' in existing_keys:
        df_ = store.get('metadata')
        assert np.all(df.index == store.get('metadata')), 'embed: The input metadata and existing metadata do not match.'
    else:
        store.put('metadata', df, format='table', data_columns=None)

    if (args.feature_type not in existing_keys) or (overwrite):
        print(f'embed: Generating embeddings for {args.feature_type}.')
        embedder = get_embedder(args.feature_type)
        embeddings = embedder(df.seq.values.tolist())
        store.put(args.feature_type, pd.DataFrame(embeddings, index=df.index), format='table', data_columns=None) 

    store.close()