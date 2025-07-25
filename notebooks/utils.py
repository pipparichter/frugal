import pandas as pd 
import numpy as np 
import re 
from src.files import GBFFFile, FASTAFile, InterProScanFile, BLASTJsonFile
import matplotlib.pyplot as plt 
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import chisquare
from scipy.stats.contingency import expected_freq
import dataframe_image as dfi

plt.rcParams['font.family'] = 'Arial'

get_gc_content = lambda nt_seq : (nt_seq.count('G') + nt_seq.count('C')) / len(nt_seq)

get_percent = lambda n, total : f'{100 * n / total:.2f}%' if (total > 0) else '0%'
get_text = lambda subscript, n, total : '$n_{' + subscript + '}$' + f' = {n} ({get_percent(n, total)})\n'

is_top_hit_hypothetical = lambda df : df.top_hit_product == 'hypothetical protein'
is_cds_conflict = lambda df : (df.conflict) & (df.top_hit_feature == 'CDS') & (~df.top_hit_pseudo)
is_antiparallel_cds_conflict = lambda df : is_cds_conflict(df) & (df.overlap_type.isin(['convergent', 'divergent']))
is_tandem_cds_conflict = lambda df : is_cds_conflict(df) & (df.overlap_type == 'tandem')
is_hypothetical_cds_conflict = lambda df : df.conflict & is_top_hit_hypothetical(df)
is_non_coding_conflict = lambda df : df.conflict & (df.top_hit_pseudo | (df.top_hit_feature != 'CDS'))
is_nested_cds_conflict = lambda df : df.conflict & ((df.top_hit_overlap == '11') | (df.query_overlap == '11')) & is_cds_conflict(df)

def apply_thresholds(results_df:pd.DataFrame, real_threshold:float=0.8, spurious_threshold:float=0.95):
    results_df['spurious'] = np.where(results_df.model_output_0 > spurious_threshold, True, False)
    results_df['real'] = np.where(results_df.model_output_1 > real_threshold, True, False)
    results_df['uncertain'] = ~results_df.real & ~results_df.spurious
    results_df['model_label'] = np.select([results_df.real, results_df.spurious.values, results_df.uncertain.values], ['real', 'spurious', 'uncertain'], default='none')
    return results_df


def get_chi_square_p_value(observed_counts_df:pd.DataFrame):
    # totals = observed_counts_df.sum(axis=1)
    # Not sure if I should be testing for independence, or using the null that they are equally-distributed. 
    # expected_counts_df = pd.DataFrame(0.5, index=observed_counts_df.index, columns=observed_counts_df.columns)
    # expected_counts_df = expected_counts_df.mul(totals, axis=0)
    expected_counts_df = pd.DataFrame(expected_freq(observed_counts_df), index=observed_counts_df.index, columns=observed_counts_df.columns) # This uses frequencies based on the marginal frequencies.
    p = chisquare(observed_counts_df.values.ravel(), expected_counts_df.values.ravel()).pvalue
    return p


def write_fasta(df:pd.DataFrame, path:str=None, add_top_hit:bool=True):
    '''Export the Prodigal-predicted and reference sequences to a FASTA file.'''
    content = ''
    for row in df.itertuples():
        seq = re.sub('X{2,}', '', row.seq)
        content += f'>{row.Index}\n{seq}\n'
        if add_top_hit:
            content += f'>{row.top_hit_protein_id}\n{row.top_hit_seq}\n'
    with open(path, 'w') as f:
        f.write(content)
    print(f'write_fasta: Wrote {len(content.split('\n')) // 2} sequences to {path}')


def get_split_axes(bottom_range:tuple, top_range:tuple, figsize:tuple=(5, 5)):

    ratio = (top_range[-1] - top_range[0]) / (bottom_range[-1] - bottom_range[0])
    fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, height_ratios=[ratio, 1])

    ax_top.set_ylim(*top_range)
    ax_bottom.set_ylim(*bottom_range)

    # Hide the spines between the two plots. 
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)  # Don't show tick labels on the top
    # ax_bottom.xaxis.tick_bottom()

    # Add diagonal lines to indicate the break. 
    d = 0.015 
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    return fig, (ax_top, ax_bottom)


def write_table(df:pd.DataFrame, path:str=None):
    if path is None:
        return 
    elif path.split('.')[-1] == 'csv':
        df.to_csv(path)
    elif path.split('.')[-1] == 'png':
        dfi.export(df, path)
    return


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


def load_ncbi_genome_metadata(genome_metadata_path='../data/dataset/ncbi_genome_metadata.tsv', taxonomy_metadata_path:str='../data/dataset/ncbi_taxonomy_metadata.tsv'):
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
    levels = ['phylum', 'superkingdom', 'kingdom', 'class', 'order', 'genus', 'species', 'family']
    col_names.update({f'{level}_taxid':f'{level}_taxonomy_id' for level in levels})
    col_names.update({f'{level}_name':f'{level}' for level in levels})
    col_names['phylum_taxid'] = 'phylum_taxonomy_id'
    col_names['class_taxid'] = 'class_taxonomy_id'
    col_names['order_taxid'] = 'order_taxonomy_id'
    col_names['genus_taxid'] = 'genus_taxonomy_id'
    col_names['species_taxid'] = 'species_taxonomy_id'
    col_names['family_taxid'] = 'family_taxonomy_id'
    col_names['kingdom_taxid'] = 'kingdom_taxonomy_id'
    col_names['assembly_accession'] = 'genome_id'

    genome_metadata_df = genome_metadata_df.rename(columns=col_names)
    genome_metadata_df = fillna(genome_metadata_df, rules={str:'none'}, errors='ignore')
    return genome_metadata_df.set_index('genome_id')




