import pandas as pd 
import numpy as np 
import re
from src import GTDB_DTYPES


def load_genome_metadata(path:str='../data/bac120_metadata_r207.tsv', reps_only:bool=True, refseq_only:bool=False):
    df = pd.read_csv(path, delimiter='\t', low_memory=False, dtype={'partial':str})
    df = df.rename(columns={'accession':'genome_id', 'ncbi_taxid':'ncbi_taxonomy_id'})

    df['prefix'] = [genome_id[:2] for genome_id in df.genome_id] 
    df['genome_id'] = [genome_id.replace('GB_', '').replace('RS_', '') for genome_id in df.genome_id] # Remove the prefixes from the genome IDs.
    df['gtdb_representative'] = [True if (is_rep == 't') else False for is_rep in df.gtdb_representative]
    df['ncbi_translation_table'] = [0 if (trans_table == 'none') else trans_table for trans_table in df.ncbi_translation_table]

    if refseq_only:
        df = df[df.prefix == 'RS']
    if reps_only:
        df = df[df.gtdb_representative]

    for level in ['phylum', 'class', 'order', 'genus', 'species', 'family', 'domain']: # Parse the taxonomy string. 
        df[level] = [re.search(f'{level[0]}__([^;]*)', taxonomy).group(1) for taxonomy in df.gtdb_taxonomy]
    df = df.drop(columns=['gtdb_taxonomy'])

    df = df[[col for col in df.columns if (col in GTDB_DTYPES.keys())]]                
    df = df.astype({col:dtype for col, dtype in GTDB_DTYPES.items() if (col in df.columns)})

    return df.set_index('genome_id')