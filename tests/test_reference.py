import unittest
from src.genome import ReferenceGenome
from src.files import FASTAFile
import glob 
from src import has_mixed_dtypes
from parameterized import parameterized


class TestReferenceGenome(unittest.TestCase):

    summary_paths = sorted(glob.glob('./data/ref/*_summary.csv'))
    results_paths = sorted(glob.glob('./data/ref/*_results.csv'))
    query_paths = sorted(glob.glob('./data/proteins/prodigal/*')) # FASTA files used as queries to the reference. 

    # The summary DataFrame should always have entries as the query FASTA file. 
    @parameterized.expand(zip(results_paths, summary_paths))
    def test_summary_and_query_ids_match(self, summary_path, results_path):
        query_ids = np.array(FASTAFile(path=query_path).ids)
        summary_ids = pd.read_csv(summary_path, index=0).index.values
        self.assertTrue(len(summary_ids) == len(query_ids), 'The number of IDs in the query and summary files are not the same.')
        self.assertTrue(np.all(query_ids == summary_ids), 'The IDs in the query and summary file are not the same.')
    
    # Searching a reference genome against itself should result in all matches.
    def test_self_search_against_same_genome_returns_all_matches(self):
        pass 

    def test_all_hits_have_a_top_hit_genome_id(self): # Adding because of a bug I encountered.
        pass 
    
    # I don't think there are any edge cases where this would not be true.
    def test_all_hits_with_aligned_start_codons_are_in_frame(self, df:pd.DataFrame):
        pass 
    # for path in paths:
    #     df = pd.read_csv(path)
    #     df = df[df.n_hits > 0]
    #     assert df.top_hit_genome_id.nunique() == 1, df.top_hit_genome_id.unique()

    def test_overlap_length_is_not_negative(self):
        pass 

    def test_overlap_is_less_than_query_length(self):
        pass 



if __name__ == '__main__':

    unittest.main()

