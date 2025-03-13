import pandas as pd 
import json 
from src import fillna
import numpy as np 
import re 

class BLASTJsonFile():

    field_map = dict()
    field_map['accession'] = 'subject_id'
    field_map['query_title'] = 'id'
    field_map['title'] = 'subject_description'
    field_map['sciname'] = 'subject_taxon'
    field_map['taxid'] = 'subject_taxonomy_id'
    field_map['bit_score'] = 'bit_score'
    field_map['evalue'] = 'e_value'
    field_map['identity'] = 'sequence_identity'
    field_map['hit_from'] = 'subject_alignment_start'
    field_map['hit_to'] = 'subject_alignment_stop'
    field_map['query_from'] = 'query_alignment_start'
    field_map['query_to'] = 'query_alignment_stop'
    field_map['gaps'] = 'n_gaps'
    field_map['align_len'] = 'alignment_length'
    field_map['qseq'] = 'query_seq'
    field_map['hseq'] = 'subject_seq'
    field_map['len'] = 'subject_length'
    field_map['query_len'] = 'query_length'

    fields = list(field_map.keys())
    minimal_fields = ['subject_id', 'e_value', 'subject_taxon', 'sequence_identity', 'subject_description', 'alignment_length', 'query_length', 'subject_length']

    prefix = 'blast'

    @staticmethod
    def is_hypothetical(df:pd.DataFrame):
        assert df.subject_description.isnull().sum() == 0, 'BLASTJsonFile.is_hypothetical: There are NaN values in the subject_description column.'
        # Sorting a boolean array in ascending order will put the True values at the end. 
        mask = df.subject_description.str.lower().str.contains('hypothetical')
        mask = mask | df.subject_description.str.lower().str.contains('uncharacterised')
        mask = mask | df.subject_description.str.lower().str.contains('uncharacterized')
        return mask 

    def __init__(self, path:str):
        with open(path, 'r') as f:
            content = json.load(f)

        self.n_queries = len(content['BlastOutput2'])

        df = []
        for query in content['BlastOutput2']:
            results = query['report']['results']['search']
            query_info = {field:value for field, value in results.items() if (field != 'hits')}

            if (len(results['hits']) == 0):
                df.append(query_info)

            for hit in results['hits']:
                hit_info = {'len':hit['len']}
                for hsp in hit['hsps']:
                    row = query_info.copy()
                    row.update(hit_info)
                    row.update(hit['description'][0]) # Only get the description for the first entry. 
                    row.update(hsp)
                    df.append(row)

        df = pd.DataFrame(df)
        df = df[list(BLASTJsonFile.field_map.keys())].rename(columns=BLASTJsonFile.field_map)
        self.df = pd.DataFrame(df).set_index('id')

    
    def to_df(self, drop_duplicates:bool=False, max_e_value:float=None, add_prefix:bool=False, use_minimal_fields:bool=True):
        
        df = self.df.copy()
        df = fillna(df, rules={str:'none'}, errors='ignore') # Fill in the empty sequences and subject IDs. 
        df['hypothetical'] = BLASTJsonFile.is_hypothetical(df)
        df['subject_description'] = [re.sub('\[(.+)\]', '', description) for description in df.subject_description] # Remove the taxon name from the sequence description. 
        # First sort by whether or not the hit is hypothetical, and then by E-value. This means selecting the first of hits for the same query sequence
        # will first prioritize all non-hypothetical hits over hypothetical hits. 
        df = df.sort_values(['hypothetical', 'e_value'])

        if max_e_value is not None:
            df = df[df.e_value < max_e_value].copy()
        if drop_duplicates:
            df = df[~df.index.duplicated(keep='first')]
            # assert len(df) == self.n_queries, f'BLASTJsonFile.to_df: The length of the de-duplicated BLAST results should be {n_queries}.'
        if use_minimal_fields:
            df = df[BLASTJsonFile.minimal_fields + ['hypothetical']].copy()
        if add_prefix:
            df = df.rename(columns={col:f'{BLASTJsonFile.prefix}_{col}' for col in df.columns})

        return df



    # def get_query_hits(self, query_id:pd.DataFrame, max_e_value:float=None):
        
    #     hits_df = self.df[seflf.df.index == query_id].copy()
    #     if max_e_value is not None:
    #         hits_df = hits_df[hits_df.evalue < max_e_value]

    #     hits = hits['subject_title'].values
    #     return list(hits)

            # missing_query_ids = self.query_ids[~np.isin(self.query_ids, df.index)]
            # if len(missing_query_ids) > 0:
            #     print(f'BLASTJsonFile.to_df: {len(missing_query_ids)} query sequences do not have any hits meeting the maximum E-value of threshold of {max_e_value}.')
            #     df_ = pd.DataFrame({'subject_id':'none'}, index=missing_query_ids)
            #     df = pd.concat([df, df_])