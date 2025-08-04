import pandas as pd 
import numpy as np 
import re 
from src.files import GBFFFile, FASTAFile, InterProScanFile, BLASTJsonFile
import matplotlib.pyplot as plt 
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import chisquare, mannwhitneyu
from scipy.stats.contingency import expected_freq
import dataframe_image as dfi
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.tools import MMSeqs
import src.reference
import os 
from IPython.display import display, HTML

# Functions and variables for generating figures and tables. 

WIDTH_MM = 85 # Half-page figure width for BMC Bioinformatics. 
WIDTH_IN = WIDTH_MM / 25.4
MAX_HEIGHT_MM = 225 # Maximum figure height for BMC Bioinformatics. 
MAX_HEIGHT_IN = MAX_HEIGHT_MM / 25.4

DPI = 300
    
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 6
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['lines.linewidth'] = 0.8
plt.rcParams['lines.markersize'] = 3
# sns.set_theme(style="whitegrid", rc={"lines.markersize": 4})

# Figure legend and title (up to 15 words for title, 300 words for legend) must appear in the manuscript file, not on the image or PDF.
# The figure file itself should include only the graphic—fully cropped, with embedded keys or labels inside the image, but without the caption text.

def get_figure(n_cols:int=1, aspect_ratio:float=4/3):
    '''Generate a figure which fits the requirements for BMC Bioinformatics.'''
    height = WIDTH_IN / aspect_ratio
    fig_size = (WIDTH_IN * n_cols, height)

    assert height < MAX_HEIGHT_IN, f'get_figure: Figure height must be less than {MAX_HEIGHT_IN}.'

    return plt.subplots(figsize=fig_size, ncols=n_cols)


def get_split_figure(bottom_range:tuple, top_range:tuple, aspect_ratio:float=4/3):

    height_ratio = (top_range[-1] - top_range[0]) / (bottom_range[-1] - bottom_range[0]) # Get ratio of top axes height to bottom axes height.
    height = WIDTH_IN / aspect_ratio
    fig_size = (WIDTH_IN, height)
    
    assert height < MAX_HEIGHT_IN, f'get_figure: Figure height must be less than {MAX_HEIGHT_IN}.'

    fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=fig_size, height_ratios=[height_ratio, 1])

    ax_top.set_ylim(*top_range)
    ax_bottom.set_ylim(*bottom_range)

    # Hide the spines between the two plots. 
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)  # Don't show tick labels on the top

    # Add diagonal lines to indicate the break. 
    d = 0.015 
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    return fig, (ax_top, ax_bottom)


def save_table(table_df:pd.DataFrame, label:str=None, path=None, pretty_print:bool=True):

    kwargs = dict()
    kwargs['index'] = True # Include the DataFrame index in the table. 
    kwargs['escape'] = True # Allow characters like % and _ to be included. 
    kwargs['column_format'] = 'l' + 'c' * (table_df.shape[1] - 1) # Specifies column alignment, left for the first column, center for the rest.
    kwargs['longtable'] = False # Disallow multipage tables. 
    kwargs['bold_rows'] = False
    kwargs['multicolumn'] = True
    kwargs['na_rep'] = 'N/A'
    kwargs['float_format'] = "%.3f"
    kwargs['label'] = label
    kwargs['header'] = True # Include column headers. 
    kwargs['buf'] = None 

    latex = table_df.to_latex(**kwargs)

    # Ensure use of booktabs for clean lines. 
    latex = latex.replace(r'\toprule', r'\hline').replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline')

    if path is not None:
        with open(path, 'w') as f:
            f.write(latex)

    if pretty_print:
        display(HTML(table_df.to_html()))
        # fig, ax = plt.subplots(figsize=(table_df.shape[1] * 2, table_df.shape[0] * 0.5))
        # ax.axis('off')
        # ax.table(cellText=table_df.values, colLabels=table_df.columns, rowLabels=table_df.index, cellLoc='center', loc='center')
        # plt.show()
    else:
        print(latex) # Also print out the LaTeX text. 


def save_figure(fig, path:str=None):
    '''Save a figure in a format suitable for BMC Bioinformatics.'''
    fig.tight_layout()

    if path is not None:
        assert os.path.splitext(path)[-1] == '.pdf', f'save_figure: Figure should be saved as a PDF.'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')

    plt.show() # Also show the figure. 



get_gc_content = lambda nt_seq : (nt_seq.count('G') + nt_seq.count('C')) / len(nt_seq)

get_percent = lambda n, total : f'{100 * n / total:.2f}%' if (total > 0) else '0%'
get_text = lambda subscript, n, total : '$n_{' + subscript + '}$' + f' = {n} ({get_percent(n, total)})\n'

is_top_hit_hypothetical = lambda df : df.top_hit_product == 'hypothetical protein'
is_top_hit_ab_initio = lambda df : df.top_hit_evidence_type == 'ab initio prediction'

is_cds_conflict = lambda df : (df.conflict) & (df.top_hit_feature == 'CDS') & (~df.top_hit_pseudo)
is_antiparallel_cds_conflict = lambda df : is_cds_conflict(df) & (df.overlap_type.isin(['convergent', 'divergent']))
is_tandem_cds_conflict = lambda df : is_cds_conflict(df) & (df.overlap_type == 'tandem')
is_hypothetical_cds_conflict = lambda df : df.conflict & is_top_hit_hypothetical(df)
is_non_coding_conflict = lambda df : df.conflict & (df.top_hit_pseudo | (df.top_hit_feature != 'CDS'))
is_nested_cds_conflict = lambda df : df.conflict & ((df.top_hit_overlap == '11') | (df.query_overlap == '11')) & is_cds_conflict(df)


def apply_thresholds(results_df:pd.DataFrame, real_threshold:float=0.8, spurious_threshold:float=0.95, model_name:str='model_v2'):
    '''Apply a double-sided threshold to prediction results and assign labels to each sequence (real, uncertain, or spurious)'''
    results_df = results_df.rename(columns={f'{model_name}_output_0':'model_output_0', f'{model_name}_output_1':'model_output_1', f'{model_name}_label':'model_label'})
    results_df['spurious'] = np.where(results_df.model_output_0 > spurious_threshold, True, False)
    results_df['real'] = np.where(results_df.model_output_1 > real_threshold, True, False)
    results_df['uncertain'] = ~results_df.real & ~results_df.spurious
    results_df['model_label'] = np.select([results_df.real, results_df.spurious, results_df.uncertain], ['real', 'spurious', 'uncertain'], default='none')
    return results_df


def load_dataset(dataset_path:str=None, exclude_genome_ids:list=None, annotate:bool=True):
    dataset_df = pd.read_csv(dataset_path, index_col=0)
    if annotate:
        dataset_df = src.reference.annotate(dataset_df)
    
    if np.any(dataset_df.columns.str.contains('top_hit')):
        dataset_df['query_length'] = dataset_df.seq.apply(len) # Make sure these are in units of amino acids. 
        dataset_df['top_hit_length'] = dataset_df.top_hit_seq.apply(len) # Make sure these are in units of amino acids. 
        # Add some columns for more straighforward analysis of conflicts. 
        dataset_df['query_codon_start'] = 1 # None of the Prodigal predictions have any offset to the translational start site.
        dataset_df['query_id'] = dataset_df.index
        dataset_df['query_product'] = 'hypothetical protein'
        dataset_df['query_seq'] = dataset_df.seq
        dataset_df['top_hit_id'] = dataset_df.top_hit_protein_id
        dataset_df['top_hit_gc_content'] = dataset_df.top_hit_nt_seq.apply(get_gc_content)

    dataset_df = dataset_df[~dataset_df.genome_id.isin(exclude_genome_ids)].copy()

    return dataset_df


def load_results(dataset_path:str=None, predictions_path:str=None, top_hit_predictions_path:str=None, exclude_genome_ids:list=['GCF_029854295.1', 'GCF_021057185.1', 'GCF_016097415.1'], annotate:bool=True):
    
    dataset_df = load_dataset(dataset_path=dataset_path, exclude_genome_ids=exclude_genome_ids, annotate=annotate)

    fields = ['real', 'spurious', 'model_output_1', 'model_output_0', 'uncertain', 'model_label']

    results_df = pd.read_csv(predictions_path, index_col=0)
    results_df = results_df.merge(dataset_df, left_index=True, right_index=True, how='inner')
    results_df = apply_thresholds(results_df, real_threshold=0.8, spurious_threshold=0.9)

    if top_hit_predictions_path is not None:
        top_hit_results_df = pd.read_csv(top_hit_predictions_path, index_col=0)
        top_hit_results_df = apply_thresholds(top_hit_results_df, real_threshold=0.8, spurious_threshold=0.9)
        top_hit_results_df = top_hit_results_df[~top_hit_results_df.index.duplicated(keep='first')].copy()
        # Add the top hit predictions to the results DataFrame. 
        with pd.option_context('future.no_silent_downcasting', True):
            for field in fields:
                results_df[f'top_hit_{field}'] = results_df.top_hit_protein_id.map(top_hit_results_df[field])
    
    # Add some columns for more straighforward analysis of conflicts. 
    for field in fields:
        results_df[f'query_{field}'] = results_df[field]
    
    return results_df


def has_alignment(df, path:str='../data/results/results-2/dataset_swissprot_align_mmseqs.tsv', min_bit_score:float=50):

    align_df = MMSeqs.load_align(path)
    alignment_ids = align_df[align_df.bit_score > min_bit_score].query_id.unique()
    df['aligned'] = df.index.isin(alignment_ids)
    return df


def has_mixed_dtypes(df:pd.DataFrame):
    n_dtypes = np.array([df[col].apply(type).nunique() for col in df.columns])
    return (n_dtypes > 1).sum() > 0


def get_dtypes(df:pd.DataFrame):
    dtypes = dict()
    for col in df.columns:
        dtype = df[col].dropna().apply(type).values
        if (len(dtype) == 0):
            warnings.warn(f'get_dtypes: Column "{col}" only contains NaNs. Inferring datatype as strings.')
            dtype = [str]
        dtypes[col] = dtype[0]
    return dtypes


def fillna(df:pd.DataFrame, rules:dict={bool:False, str:'none', int:0}, errors='raise'):
    with pd.option_context('future.no_silent_downcasting', True): # Opt-in to future pandas behavior, which will raise a warning if it tries to downcast.
        for col, dtype in get_dtypes(df).items():
            value = rules.get(dtype, None)
            if value is not None:
                df[col] = df[col].fillna(rules[dtype]).astype(dtype)
                # print(f'fillna: Replaced NaNs in column {col} with "{rules[dtype]}."')
            if errors == 'raise':
                assert df[col].isnull().sum() == 0, f'fillna: There are still NaNs in column {col}.'
    return df 



def get_pca(df:pd.DataFrame, n_components:int=2, idxs=[0, 1]):
    scaler = StandardScaler()
    values = scaler.fit_transform(df.values)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(values)
    return pd.DataFrame(components[:, idxs], index=df.index), pca.explained_variance_ratio_[idxs]



def get_chi_square_p_value(observed_counts_df:pd.DataFrame):
    # totals = observed_counts_df.sum(axis=1)
    # Not sure if I should be testing for independence, or using the null that they are equally-distributed. 
    # expected_counts_df = pd.DataFrame(0.5, index=observed_counts_df.index, columns=observed_counts_df.columns)
    # expected_counts_df = expected_counts_df.mul(totals, axis=0)
    expected_counts_df = pd.DataFrame(expected_freq(observed_counts_df), index=observed_counts_df.index, columns=observed_counts_df.columns) # This uses frequencies based on the marginal frequencies.
    p = chisquare(observed_counts_df.values.ravel(), expected_counts_df.values.ravel()).pvalue
    return p


def get_mann_whitney_p_value(*groups, n_permutations:int=100):
    '''Use a permutation test to determine the significance of the Mann-Whitney test statistic.'''
    n1, n2 = len(groups[0]), len(groups[1])
    combined = np.concatenate(list(groups)).ravel()
    mwu = mannwhitneyu(*groups, alternative='two-sided')
    stat = mwu.statistic
    
    if n_permutations is None:
        return mwu.pvalue
    
    stats = list()
    for _ in range(n_permutations):
        shuffled = np.random.permutation(combined)
        stats.append(mannwhitneyu(shuffled[:n1], shuffled[n1:n1 + n2], alternative='two-sided').statistic)
    # p = np.mean(np.array(stats) > stat)
    p = np.mean(np.abs(stats - np.median(stats)) >= np.abs(stat - np.median(stats)))
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


# Cleaning up BLAST hits for results-2-2-intergenic. 

SUBJECT_DESCRIPTION_MAP = dict()
SUBJECT_DESCRIPTION_MAP['IS630 family transposase'] = 'transposase'
SUBJECT_DESCRIPTION_MAP['IS1634 family transposase'] = 'transposase' 
SUBJECT_DESCRIPTION_MAP['IS110 family transposase'] = 'transposase'
SUBJECT_DESCRIPTION_MAP['IS481 family transposase'] = 'transposase'
SUBJECT_DESCRIPTION_MAP['Transposase'] = 'transposase'
SUBJECT_DESCRIPTION_MAP['RNA-guided endonuclease InsQ/TnpB family protein, partial'] = 'transposase' 
# SUBJECT_DESCRIPTION_MAP['helix-turn-helix domain-containing protein'] = 'transposase' # Maybe shouldn't do this. 
SUBJECT_DESCRIPTION_MAP['RNA-guided endonuclease TnpB family protein'] = 'transposase'
SUBJECT_DESCRIPTION_MAP['RNA-guided endonuclease InsQ/TnpB family protein'] = 'transposase'

SUBJECT_DESCRIPTION_MAP['integrase core domain-containing protein'] = 'integrase'
SUBJECT_DESCRIPTION_MAP['zinc ribbon domain-containing protein, partial'] = 'zinc ribbon domain-containing protein'
SUBJECT_DESCRIPTION_MAP['tyrosine-type recombinase/integrase'] = 'recombinase/integrase'
SUBJECT_DESCRIPTION_MAP['tyrosine-type recombinase/integrase, partial'] = 'recombinase/integrase'
SUBJECT_DESCRIPTION_MAP['site-specific integrase'] = 'integrase'

SUBJECT_DESCRIPTION_MAP['PP2C family serine/threonine-protein phosphatase'] = 'PP2C serine/threonine phosphatase'
SUBJECT_DESCRIPTION_MAP['PP2C family protein-serine/threonine phosphatase'] = 'PP2C serine/threonine phosphatase'
SUBJECT_DESCRIPTION_MAP['Stp1/IreP family PP2C-type Ser/Thr phosphatase'] = 'PP2C serine/threonine phosphatase'
SUBJECT_DESCRIPTION_MAP['protein phosphatase 2C domain-containing protein'] = 'PP2C serine/threonine phosphatase'

# PAS domains are positioned at the amino terminus of signaling proteins such as sensor histidine kinases, cyclic-di-GMP syntheses and hydrolases, and methyl-accepting chemotaxis proteins. 
# The protein which matches this domain also matches a histidine kinase. 
SUBJECT_DESCRIPTION_MAP['PAS domain-containing protein, partial'] = 'PAS domain-containing protein'
SUBJECT_DESCRIPTION_MAP['PAS domain S-box protein'] = 'PAS domain-containing protein'
SUBJECT_DESCRIPTION_MAP['histidine kinase dimerization/phospho-acceptor domain-containing protein'] = 'histidine kinase'

SUBJECT_DESCRIPTION_MAP['class I SAM-dependent methyltransferase'] = 'methyltransferase'
SUBJECT_DESCRIPTION_MAP['MAG: putative RNA methylase'] = 'RNA methyltransferase'
SUBJECT_DESCRIPTION_MAP['rRNA adenine dimethyltransferase family protein'] = 'RNA methyltransferase'
SUBJECT_DESCRIPTION_MAP['16S rRNA (adenine(1518)-N(6)/adenine(1519)-N(6))-dimethyltransferase RsmA'] = 'RNA methyltransferase'
SUBJECT_DESCRIPTION_MAP['16S rRNA (adenine(1518)-N(6)/adenine(1519)-N(6))-dimethyltransferase RsmA, partial'] = 'RNA methyltransferase'
SUBJECT_DESCRIPTION_MAP['16S rRNA (adenine(1518)-N(6)/adenine(1519)-N(6))-dimethyltransferase RsmA, partial'] = 'RNA methyltransferase'
SUBJECT_DESCRIPTION_MAP['dimethyltransferase'] = 'methyltransferase'

SUBJECT_DESCRIPTION_MAP['ROK family transcriptional regulator'] = 'ROK transcriptional regulator'
SUBJECT_DESCRIPTION_MAP['ROK family protein'] = 'ROK transcriptional regulator'


SUBJECT_DESCRIPTION_MAP['ATP-binding protein, partial'] = 'ATP-binding protein'

SUBJECT_DESCRIPTION_MAP['DapH/DapD/GlmU-related protein'] = 'acetyltransferase' # DapH/DapD/GlmU-related proteins are acetyltransferases.
SUBJECT_DESCRIPTION_MAP['GNAT family N-acetyltransferase'] = 'acetyltransferase'

SUBJECT_DESCRIPTION_MAP['MAG:DUF3846 domain-containing protein'] = 'DUF3846 domain-containing protein'
SUBJECT_DESCRIPTION_MAP['MAG: DUF3846 domain-containing protein'] = 'DUF3846 domain-containing protein'

SUBJECT_DESCRIPTION_MAP['DUF1361 domain-containing protein, partial'] = 'DUF1361 domain-containing protein'

SUBJECT_DESCRIPTION_MAP['MULTISPECIES: DUF1858 domain-containing protein'] = 'DUF1858 domain-containing protein'

SUBJECT_DESCRIPTION_MAP['polysaccharide pyruvyl transferase family protein'] = 'polysaccharide pyruvyl transferase'
SUBJECT_DESCRIPTION_MAP['polysaccharide pyruvyl transferase CsaB'] = 'polysaccharide pyruvyl transferase'

SUBJECT_DESCRIPTION_MAP['nucleotidyltransferase family protein'] = 'nucleotidyltransferase'

SUBJECT_DESCRIPTION_MAP['cysteine hydrolase family protein'] = 'cysteine hydrolase'

SUBJECT_DESCRIPTION_MAP['type II toxin-antitoxin system HicA family toxin'] = 'HicA toxin'

SUBJECT_DESCRIPTION_MAP['MULTISPECIES: type II toxin-antitoxin system RelE/ParE family toxin'] = 'RelE/ParE toxin'
SUBJECT_DESCRIPTION_MAP['type II toxin-antitoxin system RelE/ParE family toxin'] = 'RelE/ParE toxin'

SUBJECT_DESCRIPTION_MAP['DUF29 family protein'] = 'DUF29 domain-containing protein'
SUBJECT_DESCRIPTION_MAP['ribbon-helix-helix protein, CopG family'] = 'CopG ribon-helix-helix protein'
SUBJECT_DESCRIPTION_MAP['DUF6290 family protein'] = 'DUF6290 domain-containing protein'

SUBJECT_DESCRIPTION_MAP['hemolysin family protein'] = 'hemolysin'

SUBJECT_DESCRIPTION_MAP['type I-B CRISPR-associated protein Cas7/Cst2/DevR'] = 'CRISPR-associated protein Cas7/Cst2/DevR'
SUBJECT_DESCRIPTION_MAP['MAG: type I-B CRISPR-associated protein Cas7/Cst2/DevR'] = 'CRISPR-associated protein Cas7/Cst2/DevR'
SUBJECT_DESCRIPTION_MAP['CRISPR-associated autoregulator, Cst2 family, partial'] = 'CRISPR-associated protein Cas7/Cst2/DevR'
SUBJECT_DESCRIPTION_MAP['CRISPR-associated autoregulator, Cst2 family'] = 'CRISPR-associated protein Cas7/Cst2/DevR'

SUBJECT_DESCRIPTION_MAP['FtsX-like permease family protein'] = 'permease'
SUBJECT_DESCRIPTION_MAP['MAG: ABC transporter permease, partial'] = 'ABC transporter permease'
SUBJECT_DESCRIPTION_MAP['MAG: ABC transporter permease'] = 'ABC transporter permease'

SUBJECT_DESCRIPTION_MAP['transglutaminase domain-containing protein']= 'transglutaminase'
SUBJECT_DESCRIPTION_MAP['MULTISPECIES: transglutaminase domain-containing protein']= 'transglutaminase'

SUBJECT_DESCRIPTION_MAP['Spo0E family sporulation regulatory protein-aspartic acid phosphatase'] = 'aspartic acid phosphatase SpoE'

SUBJECT_DESCRIPTION_MAP['(2Fe-2S) ferredoxin domain-containing protein'] = 'ferredoxin domain-containing protein'

SUBJECT_DESCRIPTION_MAP['MULTISPECIES: V-type ATP synthase subunit I'] = 'ATP synthase subunit I'
SUBJECT_DESCRIPTION_MAP['V-type ATP synthase subunit I'] = 'ATP synthase subunit I'

SUBJECT_DESCRIPTION_MAP['aspartate aminotransferase family protein'] = 'aspartate aminotransferase'

SUBJECT_DESCRIPTION_MAP['cation-translocating P-type ATPase C-terminal domain-containing protein'] = 'ATPase'
SUBJECT_DESCRIPTION_MAP['cation-translocating P-type ATPase'] = 'ATPase'

SUBJECT_DESCRIPTION_MAP['glycosyltransferase family 10 domain-containing protein'] = 'glycosyltransferase'
SUBJECT_DESCRIPTION_MAP['glycosyltransferase family 10'] = 'glycosyltransferase'

SUBJECT_DESCRIPTION_MAP['aminotransferase class V-fold PLP-dependent enzyme'] = 'aminotransferase'
SUBJECT_DESCRIPTION_MAP['cysteine desulfurase family protein'] = 'cysteine desulfurase'

SUBJECT_DESCRIPTION_MAP['MULTISPECIES: flagellin lysine-N-methylase'] = 'flagellin lysine-N-methylase'

SUBJECT_DESCRIPTION_MAP["5'-nucleotidase C-terminal domain-containing protein"] = "nucleotidase"
SUBJECT_DESCRIPTION_MAP["bifunctional metallophosphatase/5'-nucleotidase"] = "nucleotidase/metallophosphatase"

SUBJECT_DESCRIPTION_MAP['amidohydrolase family protein'] = 'amidohydrolase'

# PF20097 corresponds to DUF6487, a group of proteins ~70 amino acids in length. According to InterPro, it is likely to be a zinc-binding domain. 
SUBJECT_DESCRIPTION_MAP['PF20097 family protein'] = 'zinc binding domain-containing protein'





