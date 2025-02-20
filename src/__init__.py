import numpy as np 
import torch 
import random
import re
import matplotlib.pyplot as plt 
import pandas as pd 

plt.rcParams['font.family'] = 'Arial'


def seed(seed:int=42):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_genome_id(input_path:str) -> str:

    pattern = r'GCF_([0-9]+)\.([0-9]{1,})'
    genome_id = re.search(pattern, input_path)
    return genome_id.group(0) if (genome_id is not None) else None


def get_hypothetical(df:pd.DataFrame):
    product_col = 'product' if ('product' in df.columns) else 'ref_product'
    return df.copy()[df[product_col] == 'hypothetical protein']

def get_ab_initio(df:pd.DataFrame):
    evidence_type_col = 'evidence_type' if ('evidence_type' in df.columns) else 'ref_evidence_type'
    return df.copy()[df[evidence_type_col] == 'ab initio prediction']

def get_suspect(df:pd.DataFrame):
    return get_hypothetical(get_ab_initio(df))

def remove_hypothetical(df:pd.DataFrame):
    index = get_hypothetical(df).index
    print(f'remove_hypothetical: Removing {len(index)} sequences marked as "hypothetical protein."')
    df = df.copy().drop(index=index)
    return df

def remove_ab_initio(df:pd.DataFrame):
    index = get_ab_initio(df).index
    print(f'remove_ab_initio: Removing {len(index)} sequences with only ab initio evidence."')
    df = df.copy().drop(index=index)
    return df

def remove_suspect(df:pd.DataFrame):
    index = get_suspect(df).index
    print(f'remove_suspect: Removing {len(index)} sequences which are marked as "hypothetical protein and have only ab initio evidence."')
    df = df.copy().drop(index=index)
    return df



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