import numpy as np 
import pandas as pd 
from src.build import Build 
from setuptools import Command
import argparse
from src.genome import ReferenceGenome
from src.files import FASTAFile
import re
import glob
from tqdm import tqdm 
import os
from src import get_genome_id

        

def build():
    pass 


def ref():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', nargs='+', type=str)
    parser.add_argument('--out-dir', '-o', default='../data/ref.out/', type=str)
    parser.add_argument('--ref-dir', default='../data/refseq/', type=str)
    parser.add_argument('--prodigal-output', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # pbar = tqdm(glob.glob(args.input), desc='ref: Searching reference genomes.')
    pbar = tqdm(args.input, desc='ref: Searching reference genomes.')

    for path in pbar:

        genome_id = get_genome_id(path)
        assert genome_id is not None, f'ref: Could not extract a genome ID from the input path {path}.'

        output_path = os.path.join(args.out_dir, f'{genome_id}.ref.csv')
        pbar.set_description(f'ref: Searching reference genome for {genome_id}.')
        
        if os.path.exists(output_path) and (not args.overwrite):
            continue

        ref_path = os.path.join(args.ref_dir, f'{genome_id}_genomic.gbff')
        genome = ReferenceGenome(ref_path, genome_id=genome_id)

        df = FASTAFile(path).to_df(prodigal_output=args.prodigal_output)
        results_df = genome.search(df, verbose=False)

        results_df.to_csv(output_path)

    print(f'ref: Search complete. Output written to {args.out_dir}')




