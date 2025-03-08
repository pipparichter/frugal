import pandas as pd 
import unittest 


class TestLabeler(unittest.TestCase):

    def test_one_category_assigned_per_sequence(self):
        pass 

    def test_no_autolabeled_pseudogenes(self):
        pass 

    def test_no_null_entries(self):
        pass 

    def test_all_conflict_have_at_least_one_hit(self):
        pass 
    
    # Based on how I defined "matches," there should be no minimum overlap requirement.
    def test_no_intergenic_are_in_frame(self):
        pass 







if __name__ == '__main__':
    unittest.main()