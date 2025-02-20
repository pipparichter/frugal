import pandas as pd 
import numpy as np 
import re
from src import GTDB_DTYPES
import os 
import glob
from src import *  
from src.files import GBFFFile
from tqdm import tqdm 
import warnings

def apply_line_settings(ax:plt.Axes):
    settings = dict()
    settings['false positive'] = {'color':'tab:red'}
    settings['true positive'] = {'color':'tab:green'}
    settings['false negative'] = {'color':'tab:red', 'ls':'--'}
    settings['true negative'] = {'color':'tab:green', 'ls':'--'}

    for line in ax.lines:
        if line.get_label() in settings:
            line.set(**settings[line.get_label()])


def apply_patch_settings(ax:plt.Axes):
    settings = dict()
    settings['false positive'] = {'color':'tab:red', 'edgecolor':'black'}
    settings['true positive'] = {'color':'tab:green', 'edgecolor':'black'}
    settings['false negative'] = {'color':'tab:red', 'hatch':'///', 'edgecolor':'black'}
    settings['true negative'] = {'color':'tab:green', 'hatch':'///', 'edgecolor':'black'}

    curr_label = '_nolegend_'
    for patch in ax.patches:
        if (patch.get_label() == '_nolegend_') and (curr_label != '_nolegend_'):
            patch.set(**settings[curr_label])
        elif patch.get_label() in settings:
            curr_label = patch.get_label()
            patch.set(**settings[curr_label])


def check_cds(df:pd.DataFrame):
    feature_col = 'feature' if ('feature' in df.columns) else 'ref_feature' 
    features = ', '.join([str(feature) for feature in df[feature_col].unique()])
    # assert np.all(df[feature_col] == 'CDS'), f'check_cds: Not all entries in the input DataFrame correspond to CDS features. Found: {features}'
    if not np.all(df[feature_col] == 'CDS'):
        warnings.warn(f'check_cds: Not all entries in the input DataFrame correspond to CDS features. Found: {features}')

def get_frac_hypothetical(df:pd.DataFrame):
    check_cds(df)
    return len(get_hypothetical(df)) / len(df)

def get_frac_ab_initio(df:pd.DataFrame):
    check_cds(df)
    return len(get_ab_initio(df)) / len(df)

def get_frac_suspect(df:pd.DataFrame):
    check_cds(df)
    return len(get_suspect(df)) / len(df)

def remove_partial(df:pd.DataFrame):
    mask = (df.ref_partial != '00') & (df.partial != '00')
    print(f'remove_partial: Removing {mask.sum()} instances marked as partial by both Prodigal and RefSeq from the DataFrame.')
    return df[~mask]

def remove_pseudogenes(df:pd.DataFrame):
    pseudo_col = 'pseudo' if ('pseudo' in df.columns) else 'ref_pseudo'
    mask = df[pseudo_col].astype(bool)
    print(f'remove_pseudogenes: Removing {mask.sum()} sequences marked as pseudogenes from the DataFrame.')
    return df[~mask]


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


def load_pred_out(path:str, model_name:str='filter_balance_classes_and_lengths', ref_out_df:pd.DataFrame=None):

    df = pd.read_csv(path, index_col=0)

    cols = [col for col in df.columns if ((model_name in col) or (col == 'label'))]
    df = df[cols].copy()
    df = df.rename(columns={col:col.replace(f'{model_name}', 'model') for col in cols})

    if ref_out_df is not None: # Add the ref output data to the predictions. 
        df = df.merge(ref_out_df, left_index=True, right_index=True, how='left', validate='one_to_many')

    confusion_matrix = np.where((df.model_label == 1) & (df.label == 0), 'false positive', '')
    confusion_matrix = np.where((df.model_label  == 1) & (df.label == 1), 'true positive', confusion_matrix)
    confusion_matrix = np.where((df.model_label == 0) & (df.label == 1), 'false negative', confusion_matrix)
    confusion_matrix = np.where((df.model_label  == 0) & (df.label == 0), 'true negative', confusion_matrix)
    df['confusion_matrix'] = confusion_matrix

    return df


def load_ref_out(output_dir:str='../data/ref.out', genome_metadata_df:pd.DataFrame=None, feature:str=None):

    df = []
    for path in tqdm(glob.glob(os.path.join(output_dir, '*')), desc='load_ref_out'):
        genome_id = get_genome_id(path)
        df += [pd.read_csv(path, index_col=0, dtype={'partial':str, 'ref_partial':str}).assign(genome_id=genome_id)]
    df = pd.concat(df)

    if (feature is not None): # Filter for a specific feature, if specified.
        df = df[df.ref_feature == feature]
    
    if genome_metadata_df is not None:
        df = df.merge(genome_metadata_df, left_on='genome_id', right_index=True, how='inner')
    
    return df 


def load_refseq(refseq_dir:str='../data/refseq', genome_metadata_df:pd.DataFrame=None, feature:str='CDS', load_existing:str=None):

    if (load_existing is not None):
        df = pd.read_csv(load_existing, index_col=0, low_memory=False)
    else:
        df = []
        for path in tqdm(glob.glob(os.path.join(refseq_dir, '*')), desc='load_refseq'):
            genome_id = get_genome_id(path)
            df += [GBFFFile(path).to_df().assign(genome_id=genome_id)]
        df = pd.concat(df)

    if (feature is not None): # Filter for a specific feature, if specified.
        df = df[df.feature == feature]
    
    if genome_metadata_df is not None:
        df = df.merge(genome_metadata_df, left_on='genome_id', right_index=True, how='inner')
    
    return df  
    
