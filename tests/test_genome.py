import unittest
from src.genome import ReferenceGenome
from src.files import FASTAFile
import glob 
from src import has_mixed_dtypes
from parameterized import parameterized

# TODO: Add tests to make sure the gene boundaries actually result in the provided sequence.
# TODO: Test that things that are in-frame have high sequence identity when translated. 
# TODO: Test that all pseudogenes without a homologous sequence used some non-AA seq based evidence. 

class TestReferenceGenome(unittest.TestCase):

    summary_paths = sorted(glob.glob('./data/ref/*_summary.csv'))
    results_paths = sorted(glob.glob('./data/ref/*_results.csv'))
    query_paths = sorted(glob.glob('./data/'))

    # The summary DataFrame should always have the same length as the query DataFrame. 
    @parameterized.expand(zip(results_paths, summary_paths))
    def test_summary_and_query_have_same_size(self, summary_path, results_path):
        pass 
    
    # Searching a reference genome against itself should result in all matches.
    def test_self_search_returns_all_matches(self):
        pass 
    
    # I don't think there are any edge cases where this would not be true.
    def test_all_hits_with_aligned_start_codons_are_in_frame(self, df:pd.DataFrame):
        pass 

    def test_n_valid_hits_no_larger_than_n_hits(self):
        pass 

    def test_n_valid_hits_no_larger_than_n_hits_same_strand(self):
        pass

    def test_n_valid_hits_is_no_greater_than_one(self):
        pass  

    def test_overlap_length_is_not_negative(self):
        pass 

    def test_overlap_is_less_than_query_length(self):
        pass 

    def test_valid_hit_overlap_divisible_by_three_unless_partial(self):
        pass 

if __name__ == '__main__':

    unittest.main()

