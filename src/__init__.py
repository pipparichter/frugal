import numpy as np 
import re
import pandas as pd 
import warnings 
from Bio.Align import PairwiseAligner
import itertools

# plt.rcParams['font.family'] = 'Arial'


def get_genome_id(input_path:str, errors='raise', default:str=None) -> str:

    pattern = r'GCF_([0-9]+)\.([0-9]{1,})'
    genome_id = re.search(pattern, input_path)
    if (genome_id is None) and (errors == 'raise'):
        raise Exception(f'get_genome_id: Could not extract a genome ID from {input_path}.')
    return genome_id.group(0) if (genome_id is not None) else default


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
        for col, dtype in get_dtypes(df, errors=errors).items():
            value = rules.get(dtype, None)
            if value is not None:
                df[col] = df[col].fillna(rules[dtype]).astype(dtype)
                # print(f'fillna: Replaced NaNs in column {col} with "{rules[dtype]}."')
            if errors == 'raise':
                assert df[col].isnull().sum() == 0, f'fillna: There are still NaNs in column {col}.'
    return df 


def get_sequence_identity(seqs:list) -> float: 
    aligner = PairwiseAligner(match_score=1, mismatch_score=0, gap_score=0)
    alignment = aligner.align(seqs[0], seqs[1])[0] # I think this will get the best alignment?
    score = alignment.score
    score = max(score / len(seqs[0]), score / len(seqs[1])) # Normalize the score by sequence length. 
    return score

def get_sequence_identities(seq_df:pd.DataFrame, as_matrix:bool=False):
    n = len(seq_df)

    if n == 1:
        return np.ones(1) if (not as_matrix) else np.ones((1, 1))
    
    idxs = itertools.combinations(seq_df.index, 2)
    idxs = [(i, j) for i, j in idxs if (i != j)]

    if (not as_matrix):
        return np.array([get_sequence_identity((seq_df.loc[i].seq, seq_df.loc[j].seq)) for i, j in idxs])
    
    values = pd.DataFrame(np.eye(n), index=seq_df.index, columns=seq_df.index)
    for i, j in idxs:
        sequence_identity = get_sequence_identity((seq_df.loc[i].seq, seq_df.loc[j].seq))
        values.loc[i, j] = sequence_identity
        values.loc[j, i] = sequence_identity
    return values



GTDB_DTYPES = dict()
GTDB_DTYPES['description'] = str
GTDB_DTYPES['seq'] = str
GTDB_DTYPES['start'] = int
GTDB_DTYPES['stop'] = int
GTDB_DTYPES['strand'] = int
GTDB_DTYPES['ID'] = str
GTDB_DTYPES['partial'] = str
GTDB_DTYPES['start_type'] = str
GTDB_DTYPES['rbs_motif'] = str
GTDB_DTYPES['rbs_spacer'] = str
GTDB_DTYPES['gc_content'] = float
GTDB_DTYPES['genome_id'] = str
GTDB_DTYPES['ambiguous_bases'] = int
GTDB_DTYPES['checkm_completeness'] = float
GTDB_DTYPES['checkm_contamination'] = float
GTDB_DTYPES['checkm_marker_count'] = int
GTDB_DTYPES['checkm_marker_lineage'] = str
GTDB_DTYPES['checkm_marker_set_count'] = int
GTDB_DTYPES['coding_bases'] = int
GTDB_DTYPES['coding_density'] = float
GTDB_DTYPES['contig_count'] = int
GTDB_DTYPES['gc_count'] = int
GTDB_DTYPES['gc_percentage'] = float
GTDB_DTYPES['genome_size'] = int
GTDB_DTYPES['gtdb_genome_representative'] = str 
GTDB_DTYPES['gtdb_representative'] = bool
GTDB_DTYPES['gtdb_taxonomy'] = str 
GTDB_DTYPES['l50_contigs'] = int
GTDB_DTYPES['l50_scaffolds'] = int
GTDB_DTYPES['longest_contig'] = int
GTDB_DTYPES['longest_scaffold'] = int
GTDB_DTYPES['mean_contig_length'] = int
GTDB_DTYPES['mean_scaffold_length'] = int
GTDB_DTYPES['mimag_high_quality'] = str
GTDB_DTYPES['mimag_low_quality'] = str
GTDB_DTYPES['mimag_medium_quality'] = str
GTDB_DTYPES['n50_contigs'] = int
GTDB_DTYPES['n50_scaffolds'] = int
GTDB_DTYPES['ncbi_assembly_level'] = str
GTDB_DTYPES['ncbi_bioproject'] = str
GTDB_DTYPES['ncbi_biosample'] = str
GTDB_DTYPES['ncbi_genbank_assembly_accession'] = str
GTDB_DTYPES['ncbi_genome_category'] = str
GTDB_DTYPES['ncbi_refseq_category'] = str
GTDB_DTYPES['ncbi_genome_representation'] = str
GTDB_DTYPES['ncbi_isolate'] = str
GTDB_DTYPES['ncbi_isolation_source'] = str
GTDB_DTYPES['ncbi_translation_table'] = int
GTDB_DTYPES['ncbi_species_taxid'] = int
GTDB_DTYPES['ncbi_taxid'] = int
GTDB_DTYPES['ncbi_taxonomy_id'] = int
GTDB_DTYPES['ncbi_taxonomy'] = str
GTDB_DTYPES['protein_count'] = int
GTDB_DTYPES['scaffold_count'] = int
GTDB_DTYPES['total_gap_length'] = int
GTDB_DTYPES['trna_aa_count'] = int
GTDB_DTYPES['trna_count'] = int
GTDB_DTYPES['trna_selenocysteine_count'] = int
GTDB_DTYPES['phylum'] = str
GTDB_DTYPES['class'] = str
GTDB_DTYPES['order'] = str
GTDB_DTYPES['genus'] = str
GTDB_DTYPES['species'] = str
GTDB_DTYPES['family'] = str
GTDB_DTYPES['domain'] = str
GTDB_DTYPES['prefix'] = str