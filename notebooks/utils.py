import pandas as pd 
import numpy as np 
import re
from src import GTDB_DTYPES
import os 
import glob
from src import *  
from src.files import GBFFFile
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression

plt.rcParams['font.family'] = 'Arial'

is_n_trunc = lambda row : ((row.start > row.ref_start) and (row.strand == 1)) or ((row.stop < row.ref_stop) and (row.strand == -1)) 
is_c_trunc = lambda row : ((row.stop < row.ref_stop) and (row.strand == 1)) or ((row.start > row.ref_start) and (row.strand == -1)) 
is_n_ext = lambda row : ((row.start < row.ref_start) and (row.strand == 1)) or ((row.stop > row.ref_stop) and (row.strand == -1)) 
is_c_ext = lambda row : ((row.stop > row.ref_stop) and (row.strand == 1)) or ((row.start < row.ref_start) and (row.strand == -1)) 

get_col = lambda row, col : f'ref_{col}' if (f'ref_{col}' in row.index) else col

is_hypothetical = lambda row : row[get_col(row, 'product')] == 'hypothetical protein'
is_ab_initio = lambda row : (row[get_col(row, 'evidence_type')] == 'ab initio prediction') or (pd.isnull(row[get_col(row, 'evidence_type')]))

is_putative_ncbi = lambda row : is_hypothetical(row) and is_ab_initio(row)
is_putative_prodigal = lambda row : (row.n_valid_hits == 0)  # Assuming all hits are for coding regions. 

has_interpro_hit = lambda row : not (pd.isnull(row.interpro_analysis))

is_validated = lambda row : (not is_putative_ncbi(row)) and (not is_putative_prodigal(row))

# is_ncbi_error = lambda row : (is_putative(row)) and (not has_interpro_hit(row))
# is_prodigal_error = lambda row : (has_ref_hit(row) and is_ncbi_error(row)) or ((not has_interpro_hit(row)) and (not has_ref_hit(row)))


def remove_partial(df:pd.DataFrame):
    assert (df.partial.isnull().sum() == 0), 'remove_partial: Some of the proteins do not have a partial indicator.'
    assert (df.partial.dtype == 'object') and (df.ref_partial.dtype == 'object'), 'remove_partial: It seems as though the partial indicators were not loaded in as strings.'
    mask = ((df.partial != '00') & pd.isnull(df.ref_partial))
    mask = mask | ((df.partial != '00') & (df.ref_partial != '00')) 
    print(f'remove_partial: Removing {int(mask.sum())} sequences marked as partial by both Prodigal and the reference.')
    return df[~mask].copy()


def get_lengths(df:pd.DataFrame, ref:bool=True):
    start_col, stop_col = ('ref_' if ref else '') +'start', ('ref_' if ref else '') + 'stop'
    lengths = df[stop_col] - df[start_col] 
    lengths = lengths // 3 + 1 # Convert to amino acid units. 
    if pd.isnull(lengths).sum() > 0:
        warnings.warn('get_lengths: Some of the returned lengths are NaNs, which probably means there are sequences that do not have NCBI reference hits.')
    return lengths


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


def load_genome_metadata(genome_metadata_path='../data/ncbi_genome_metadata.tsv', taxonomy_metadata_path:str='../data/ncbi_taxonomy_metadata.tsv'):
    taxonomy_metadata_df = pd.read_csv(taxonomy_metadata_path, delimiter='\t', low_memory=False)
    genome_metadata_df = pd.read_csv(genome_metadata_path, delimiter='\t', low_memory=False)

    taxonomy_metadata_df = taxonomy_metadata_df.drop_duplicates('Taxid')
    genome_metadata_df = genome_metadata_df.drop_duplicates('Assembly Accession')

    df = genome_metadata_df.merge(taxonomy_metadata_df, right_on='Taxid', left_on='Organism Taxonomic ID', how='left')
    df = df.rename(columns={col:'_'.join(col.lower().split()) for col in df.columns})
    df = df.rename(columns={'assembly_accession':'genome_id'})
    df = fillna(df, rules={str:'none'}, check=False)
    return df.set_index('genome_id')


def load_pred_out(path:str, model_name:str=''):

    df = pd.read_csv(path, index_col=0)

    cols = [col for col in df.columns if ((model_name in col) or (col == 'label'))]
    df = df[cols].copy()
    df = df.rename(columns={col:col.replace(f'{model_name}', 'model') for col in cols})

    confusion_matrix = np.where((df.model_label == 1) & (df.label == 0), 'false positive', '')
    confusion_matrix = np.where((df.model_label  == 1) & (df.label == 1), 'true positive', confusion_matrix)
    confusion_matrix = np.where((df.model_label == 0) & (df.label == 1), 'false negative', confusion_matrix)
    confusion_matrix = np.where((df.model_label  == 0) & (df.label == 0), 'true negative', confusion_matrix)
    df['confusion_matrix'] = confusion_matrix
    df['model_name'] = model_name

    return df


def load_ref_out(output_dir:str='../data/ref.out'):

    dtypes = {'partial':str, 'ref_partial':str}
    paths = glob.glob(os.path.join(output_dir, '*'))
    df = pd.concat([pd.read_csv(path, index_col=0, dtype=dtypes).assign(genome_id=get_genome_id(path)) for path in paths])
    return df 

    
    # if genome_metadata_df is not None:
    #     genome_ids = np.intersect1d(genome_metadata_df.index, df.genome_id.unique())
    #     if len(genome_ids) < df.genome_id.nunique():
    #         warnings.warn(f'load_ref_out: Merging the genome metadata will drop ref output for {df.genome_id.nunique() - len(genome_ids)} genomes.')
    #     df = df.merge(genome_metadata_df, left_on='genome_id', right_index=True, how='inner')
    