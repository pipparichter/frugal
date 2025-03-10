import pandas as pd 
import json 
from src import fillna
import numpy as np 

class BLASTJsonFile():

    field_map = dict()
    field_map['accession'] = 'subject_id'
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
    field_map['qseq'] = 'query_seq'
    field_map['hseq'] = 'subject_seq'

    fields = list(field_map.keys())
    minimal_fields = ['subject_id', 'e_value', 'subject_taxon', 'sequence_identity']

    def __init__(self, path:str):
        with open(path, 'r') as f:
            content = json.load(f)

        self.no_hits = []
        self.n_queries = len(content['BlastOutput2'])

        df = []
        for query in content['BlastOutput2']:
            results = query['report']['results']['search']
            query_id = results['query_title']
            if (len(results['hits']) == 0):
                df.append({'query_id':query_id, 'evalue':np.inf})
                self.no_hits.append(query_id)
                # print(f'BLASTJsonFile.to_df: No hits for query {query_id}.')
            for hit in results['hits']:
                for hsp in hit['hsps']:
                    row = {'query_id':query_id}
                    row.update(hit['description'][0]) # Only get the description for the first entry. 
                    row.update(hsp)
                    df.append(row)
        self.df = pd.DataFrame(df).set_index('query_id')
        self.query_ids = np.array(list(self.df.index))

    def get_query_hits(self, query_id:pd.DataFrame, max_e_value:float=None):
        
        hits_df = self.df[seflf.df.index == query_id].copy()
        if max_e_value is not None:
            hits_df = hits_df[hits_df.evalue < max_e_value]

        hits = hits['subject_title'].values
        return list(hits)
    
    def to_df(self, drop_duplicates:bool=False, max_e_value:float=None, add_prefix:bool=False, usecols=['subject_id', 'subject_description', '']):

        df = self.df[list(BLASTJsonFile.field_map.keys())].rename(columns=BLASTJsonFile.field_map)
        df = df.sort_values('e_value')

        if max_e_value is not None:
            df = df[df.e_value < max_e_value].copy()
            # missing_query_ids = self.query_ids[~np.isin(self.query_ids, df.index)]
            # if len(missing_query_ids) > 0:
            #     print(f'BLASTJsonFile.to_df: {len(missing_query_ids)} query sequences do not have any hits meeting the maximum E-value of threshold of {max_e_value}.')
            #     df_ = pd.DataFrame({'subject_id':'none'}, index=missing_query_ids)
            #     df = pd.concat([df, df_])

        if drop_duplicates:
            df = df[~df.index.duplicated(keep='first')]
            assert len(df) == self.n_queries, f'BLASTJsonFile.to_df: The length of the de-duplicated BLAST results should be {n_queries}.'
        
        df = fillna(df, rules={str:'none'}, errors='ignore')
        if add_prefix:
            df = 
        return df