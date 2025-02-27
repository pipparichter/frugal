import unittest
from src.genome import ReferenceGenome
from src.files import FASTAFile
import glob 
from src import has_mixed_dtypes
from parameterized import parameterized


class TestReferenceGenome(unittest.Test):

    summary_paths = sorted(glob.glob('./data/ref/*_summary.csv'))
    results_paths = sorted(glob.glob('./data/ref/*_results.csv'))
    query_paths = sorted(glob.glob('./data/'))

    # The summary DataFrame should always have the same length as the query DataFrame. 
    parameterized.expand(zip(TestReferenceGenome.results_paths, TestReferenceGenome.summary_paths))
    def test_summary_and_query_have_same_size(self, summary_path, results_path):
        pass 
    
    # Searching a reference genome against itself should result in all matches.
    def test_self_search_returns_all_matches(self):
        pass 

    def test_all_start_aligned_hits_are_in_frame(self, df:pd.DataFrame):
        pass 

    def test_all_stop_aligned_hits_are_in_frame(self, df:pd.DataFrame):
        pass

    def test_n_valid_hits_no_larger_than_n_hits(self):
        pass 

    def test_n_valid_hits_no_larger_than_n_hits_same_strand(self):
        pass

    def test_n_valid_hits_no_larger_than_n_hits_in_frame(self):
        pass  

    def test_overlap_length_is_not_negative(self):
        pass 

if __name__ == '__main__':

    unittest.main()

