import unittest
from src.files import GBFFFile
from parameterized import parameterized
import os 
import glob
from src import get_genome_id
from src import has_mixed_dtypes
import numpy as np 
import pandas as pd

# TODO: Test to make sure that every CDS has a "codon_start" qualifier. 
# TODO: Add tests to make sure the gene boundaries actually result in the provided sequence.
# TODO: Test to make sure every sequence in the same genome uses the same translation table. 

class TestGBFFFile(unittest.TestCase):

    files = [GBFFFile(path) for path in glob.glob('../data/proteins/ncbi/*')[:100]]

    genome_metadata_df = pd.read_csv('../data/ncbi_genome_metadata.tsv', sep='\t')

    pseudogene_counts = dict(zip(genome_metadata_df['Assembly Accession'], genome_metadata_df['Annotation Count Gene Pseudogene']))
    coding_gene_counts = dict(zip(genome_metadata_df['Assembly Accession'], genome_metadata_df['Annotation Count Gene Protein-coding']))
    noncoding_gene_counts = dict(zip(genome_metadata_df['Assembly Accession'], genome_metadata_df['Annotation Count Gene Non-coding']))
    nucleotide_counts = dict(zip(genome_metadata_df['Assembly Accession'], genome_metadata_df['Assembly Stats Total Sequence Length']))
    contig_counts = dict(zip(genome_metadata_df['Assembly Accession'], genome_metadata_df['Assembly Stats Number of Contigs']))

    @parameterized.expand(files)
    def test_contigs_are_all_nucleotides(self, file):
        for contig in file.contigs.values():
            observed = set(list(contig))
            self.assertTrue(len(observed) <= 5, f'Failed on {file.path}. Expected to see only A,T,C,G,N but found {len(observed)} characters, {observed}.')

    # @parameterized.expand(files)
    # def test_correct_number_of_contigs(self, file):
    #     genome_id = get_genome_id(file.path)
    #     n_expected = TestGBFFFile.contig_counts[genome_id] 
    #     n_observed = len(file.contigs)
    #     self.assertTrue(n_expected == n_observed, f'Failed on {file.path}. Saw {n_observed}, expected {n_expected}.')

    @parameterized.expand(files)
    def test_correct_number_of_coding_genes(self, file):
        genome_id = get_genome_id(file.path)
        df = file.df.drop_duplicates(subset=['coordinate', 'contig_id']) # Account for the cases in which a feature has multiple entries due to a join coordinate.
        n_expected = TestGBFFFile.coding_gene_counts[genome_id] 
        n_observed = ((df.feature == 'CDS') & ~df.pseudo).sum()
        self.assertTrue(n_expected == n_observed, f'Failed on {file.path}. Saw {n_observed}, expected {n_expected}.')

    @parameterized.expand(files)
    def test_correct_number_of_pseudogenes(self, file):
        genome_id = get_genome_id(file.path)
        df = file.df.drop_duplicates(subset=['coordinate', 'contig_id'], keep='first') # Account for the cases in which a feature has multiple entries due to a join coordinate.
        n_expected = TestGBFFFile.pseudogene_counts[genome_id] 
        n_observed = (df.pseudo & (df.feature == 'CDS')).sum() # Why am I overestimating the number of pseudogenes? Ah, because RNA genes can also be marked pseudo.
        self.assertTrue(n_expected == n_observed, f'Failed on {file.path}. Saw {n_observed}, expected {n_expected}.')

    @parameterized.expand(files)
    def test_correct_number_of_noncoding_genes(self, file):
        genome_id = get_genome_id(file.path)
        df = file.df.drop_duplicates(subset=['coordinate', 'contig_id']) # Account for the cases in which a feature has multiple entries due to a join coordinate.
        n_expected = TestGBFFFile.noncoding_gene_counts[genome_id] 
        n_observed = (df.feature.isin(GBFFFile.noncoding_features)).sum()
        self.assertTrue(n_expected == n_observed, f'Failed on {file.path}. Saw {n_observed}, expected {n_expected}.')

    @parameterized.expand(files)
    def test_no_mixed_datatypes(self, file):
        self.assertTrue(not has_mixed_dtypes(file.df))

    # All CDS non-pseudo features should have a corresponding sequence.
    @parameterized.expand(files)
    def test_all_non_pseudo_cds_have_seq(self, file):
        df = file.df[(file.df.feature == 'CDS') & ~file.df.pseudo]
        self.assertTrue(np.all(df.seq != 'none'))

    # All CDS features should have a locus tag, regardless of whether or not they are a pseudogene.
    @parameterized.expand(files)
    def test_all_cds_have_locus_tag(self, file):
        df = file.df[file.df.feature == 'CDS']
        self.assertTrue(np.all(df.locus_tag != 'none'), f'Failed on {file.path}.')

    # All CDS features should have a protein ID, as long as they are not a pseudogene.
    @parameterized.expand(files)
    def test_all_cds_have_protein_id(self, file):
        df = file.df[(file.df.feature == 'CDS') & (~file.df.pseudo)]
        self.assertTrue(np.all(df.protein_id != 'none'))

    # All CDS features should have a product, regardless of whether or not they are a pseudogene.
    @parameterized.expand(files)
    def test_all_cds_have_product(self, file):
        df = file.df[file.df.feature == 'CDS']
        self.assertTrue(np.all(df.product != 'none'))
     
    @parameterized.expand(files)
    def test_no_psuedo_cds_have_seq(self, file):
        df = file.df[(file.df.feature == 'CDS') & file.df.pseudo]
        self.assertTrue(np.all(df.seq == 'none'))

    # The index column should be unique, and have the same length as the number of entries in the file. Also expect the entire integer range. 
    @parameterized.expand(files)
    def test_unique_index(self, file):
        index = file.df.index
        self.assertTrue(np.all(index == np.arange(len(index))))

    @parameterized.expand(files)
    def test_all_seq_have_translation_table(self, file):
        df = file.df[file.df.seq != 'none']
        self.assertTrue(np.all(df.translation_table != 'none'))

    # If an experiment is provided, it should always be listed as the evidence type.
    @parameterized.expand(files)
    def test_experiment_is_used_as_evidence(self, file):
        df = file.df[file.df.experiment != 'none']
        self.assertTrue(np.all(df.evidence_type == 'experiment'), f'Failed on {file.path}.')

    
        

    





if __name__ == '__main__':

    unittest.main()

