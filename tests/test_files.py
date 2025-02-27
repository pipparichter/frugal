import unittest
from src.files import GBFFFile
from parameterized import parameterized
import os 
import glob
from src import has_mixed_dtypes



class TestGBFFFile(unittest.TestCase):

    files = [GBFFFile(path) for path in glob.glob('../data/proteins/ncbi/*')][:2]

    feature_counts = dict()
    # feature_counts[]

    @parameterized.expand(files)
    def test_no_mixed_datatypes(self, file):
        self.assertTrue(not has_mixed_dtypes(file.df))

    # All CDS non-pseudo features should have a corresponding sequence.
    @parameterized.expand(files)
    def test_all_non_pseudo_cds_have_seq(self, file):
        df = file.df[(file.df.feature == 'CDS') and ~file.df.pseudo]
        self.assertTrue(np.all(df.seq != 'none'))

    # All CDS features should have a locus tag, regardless of whether or not they are a pseudogene.
    @parameterized.expand(files)
    def test_all_cds_have_locus_tag(self, file):
        df = file.df[file.df.feature == 'CDS']
        self.assertTrue(np.all(df.locus_tag != 'none'))

    # All CDS features should have a protein ID, regardless of whether or not they are a pseudogene.
    @parameterized.expand(files)
    def test_all_cds_have_protein_id(self, file):
        df = file.df[file.df.feature == 'CDS']
        self.assertTrue(np.all(df.protein_id != 'none'))

    # All CDS features should have a product, regardless of whether or not they are a pseudogene.
    @parameterized.expand(files)
    def test_all_cds_have_product(self, file):
        df = file.df[file.df.feature == 'CDS']
        self.assertTrue(np.all(df.product != 'none'))
     
    @parameterized.expand(files)
    def test_no_psuedo_cds_have_seq(self, file):
        df = file.df[(file.df.feature == 'CDS') and file.df.pseudo]
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
        self.assertTrue(np.all(df.evidence_type == 'experiment'))

    
        

    





if __name__ == '__main__':

    unittest.main()

