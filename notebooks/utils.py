import pandas as pd 
import numpy as np 
import re
from src import GTDB_DTYPES
import os 
import glob
from src import *  
from src.files import GBFFFile, FASTAFile
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from Bio.Align import PairwiseAligner

plt.rcParams['font.family'] = 'Arial'

is_n_truncated = lambda df : ((df.query_start > df.top_hit_start) & (df.query_strand == 1)) | ((df.query_stop < df.top_hit_stop) & (df.query_strand == -1)) 
is_c_truncated = lambda df : ((df.query_stop < df.top_hit_stop) & (df.query_strand == 1)) | ((df.query_start > df.top_hit_start) & (df.query_strand == -1)) 
is_n_extended = lambda df : ((df.query_start < df.top_hit_start) & (df.query_strand == 1)) | ((df.query_stop > df.top_hit_stop) & (df.query_strand == -1)) 
is_c_extended = lambda df : ((df.query_stop > df.top_hit_stop) & (df.query_strand == 1)) | ((df.query_start < df.top_hit_start) & (df.query_strand == -1)) 



def recall(df:pd.DataFrame, class_:int=0, threshold:float=0.5) -> float:
    model_labels = (df[f'model_output_{class_}'] > threshold).astype(int)
    n = ((model_labels == class_) & (df.label == class_)).sum()
    N = (df.label == class_).sum() # Total number of relevant instances (i.e. members of the class)
    return n / N

def precision(df:pd.DataFrame, class_:int=0, threshold:float=0.5) -> float:
    model_labels = (df[f'model_output_{class_}'] > threshold).astype(int)
    n = ((model_labels == class_) & (df.label == class_)).sum()
    N = (model_labels == class_).sum() # Total number of retrieved instances (i.e. predicted members of the class)
    return n / N


def get_lengths(df:pd.DataFrame, top_hit:bool=True, units='aa'):
    start_col, stop_col = ('top_hit_' if top_hit else 'query_') +'start', ('top_hit_' if top_hit else 'query_') + 'stop'
    lengths = (df[stop_col] - (df[start_col] + 1)) # The start and stop are both inclusive, so add one to the length. 

    if np.any((lengths % 3) != 0):
        warnings.warn('get_lengths: Not all gene lengths are divisible by three.')
    if pd.isnull(lengths).sum() > 0:
        warnings.warn('get_lengths: Some of the returned lengths are NaNs, which probably means there are sequences that do not have NCBI reference hits.')

    return lengths // 3 if (units == 'aa') else lengths


def denoise(df:pd.DataFrame, x_col:str=None, y_cols:list=None, bins:int=50):
    bin_labels, bin_edges = pd.cut(df[x_col], bins=bins, retbins=True, labels=False)
    df['bin_label'] = bin_labels 
    df_ = dict()
    df_[x_col] = df.groupby('bin_label', sort=True)[x_col].mean()
    for y_col in y_cols:
        df_[y_col] = df.groupby('bin_label', sort=True)[y_col].mean()
        df_[f'{y_col}_err'] = df.groupby('bin_label').apply(lambda df : df[y_col].std() / np.sqrt(len(df)), include_groups=False)
    df_ = pd.DataFrame(df_, index=df.bin_label.sort_values().unique())
    return df_


def correlation(x, y):
    linreg = LinearRegression().fit(x.reshape(-1, 1), y) 
    r2 = linreg.score(x.reshape(-1, 1), y)
    return np.round(r2, 3), linreg


def partial_correlation(x, y, z):
    # Standardize the input arrays. 
    x, y, z = (x - x.mean()) / x.std(), (y - y.mean()) / y.std(), (z - z.mean()) / z.std()

    _, linreg_zy = correlation(z, y)
    _, linreg_zx = correlation(z, x)
    # Do not need to standardize the residuale (not sure if I completely understand why)
    x_residuals = x - linreg_zx.predict(z.reshape(-1, 1))
    y_residuals = y - linreg_zy.predict(z.reshape(-1, 1))

    r2, linreg_xy = correlation(x_residuals, y_residuals)
    return r2, linreg_xy, (x_residuals, y_residuals)


def load_ncbi_genome_metadata(genome_metadata_path='../data/ncbi_genome_metadata.tsv', taxonomy_metadata_path:str='../data/ncbi_taxonomy_metadata.tsv'):
    taxonomy_metadata_df = pd.read_csv(taxonomy_metadata_path, delimiter='\t', low_memory=False)
    genome_metadata_df = pd.read_csv(genome_metadata_path, delimiter='\t', low_memory=False)

    taxonomy_metadata_df = taxonomy_metadata_df.drop_duplicates('Taxid')
    genome_metadata_df = genome_metadata_df.drop_duplicates('Assembly Accession')

    genome_metadata_df = genome_metadata_df.merge(taxonomy_metadata_df, right_on='Taxid', left_on='Organism Taxonomic ID', how='left')
    genome_metadata_df = genome_metadata_df.drop(columns=['Organism Taxonomic ID']) # This column is redundant now. 
    genome_metadata_df = genome_metadata_df.rename(columns={col:'_'.join(col.lower().split()) for col in genome_metadata_df.columns})

    col_names = dict()
    col_names['annotation_count_gene_protein-coding'] = 'n_gene_protein_coding'
    col_names['annotation_count_gene_non-coding'] = 'n_gene_non_coding'
    col_names['annotation_count_gene_pseudogene'] = 'n_pseudogene'
    col_names['assembly_stats_gc_percent'] = 'gc_percent'
    col_names['assembly_stats_total_sequence_length'] = 'total_sequence_length'
    col_names['assembly_stats_number_of_contigs'] = 'n_contigs'
    col_names['taxid'] = 'taxonomy_id'
    levels = ['phylum', 'superkingdom', 'kingdom', 'class', 'order', 'genus', 'species']
    col_names.update({f'{level}_taxid':f'{level}_taxonomy_id' for level in levels})
    col_names.update({f'{level}_name':f'{level}' for level in levels})
    col_names['phylum_taxid'] = 'phylum_taxonomy_id'
    col_names['class_taxid'] = 'class_taxonomy_id'
    col_names['order_taxid'] = 'order_taxonomy_id'
    col_names['genus_taxid'] = 'genus_taxonomy_id'
    col_names['species_taxid'] = 'species_taxonomy_id'
    col_names['kingdom_taxid'] = 'kingdom_taxonomy_id'
    col_names['assembly_accession'] = 'genome_id'

    genome_metadata_df = genome_metadata_df.rename(columns=col_names)
    genome_metadata_df = fillna(genome_metadata_df, rules={str:'none'}, errors='ignore')
    return genome_metadata_df.set_index('genome_id')


def load_predict(path:str, model_name:str=None):
    df = pd.read_csv(path, index_col=0)
    if model_name is not None:
        cols = [col for col in df.columns if ((model_name in col) or (col == 'label'))]
        df = df[cols].copy()
        df = df.rename(columns={col:col.replace(f'{model_name}', 'model') for col in cols})
        df['model_name'] = model_name
    return df


def load_labels(genome_ids:list=None, labels_dir='../data/labels'):
    paths = [os.path.join(labels_dir, f'{genome_id}_label.csv') for genome_id in genome_ids] if (genome_ids is not None) else glob.glob(os.path.join(labels_dir, '*'))
    labels_df = pd.concat([pd.read_csv(path, index_col=0) for path in paths])

    assert labels_df.index.duplicated().sum() == 0, 'load_labels: There are duplicate entries in the labels DataFrame.'
    return labels_df


def load_ref(genome_ids:list=None, ref_dir:str='../data/ref', add_labels:bool=True):
    paths = [os.path.join(ref_dir, f'{genome_id}_summary.csv') for genome_id in genome_ids] if (genome_ids is not None) else glob.glob(os.path.join(ref_dir, '*_summary.csv'))
    # Can't rely on the top_hit_genome_id column for the genome IDs, because if there is no hit it is not populated.
    dtypes = {'top_hit_partial':str, 'query_partial':str, 'top_hit_translation_table':str, 'top_hit_codon_start':str}
    ref_df = pd.concat([pd.read_csv(path, index_col=0, dtype=dtypes).assign(genome_id=get_genome_id(path)) for path in paths], ignore_index=False)
    assert ref_df.index.nunique() == len(ref_df), 'load_ref: There are duplicate entries in the ref output DataFrame.'
    if add_labels:
        labels_df = load_labels(genome_ids)
        assert len(labels_df) == len(ref_df), 'load_ref: Expected the labels and reference output DataFrames to be the same size.'
        ref_df = ref_df.merge(labels_df, right_index=True, left_index=True, validate='one_to_one', how='left')
    return ref_df
