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
    field_map['identity'] = 'identity'
    field_map['positive'] = 'positive'
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
    field_map['midline'] = 'alignment'

    fields = list(field_map.keys()) + ['n_hits', 'n_subject_taxa']
    minimal_fields = ['n_hits', 'n_subject_taxa', 'alignment', 'subject_id', 'e_value', 'subject_taxon', 'identity', 'positive', 'subject_description', 'alignment_length', 'query_length', 'subject_length']

    prefix = 'blast'

    def __init__(self, path:str):
        with open(path, 'r') as f:
            content = json.load(f)

        self.n_queries = len(content['BlastOutput2'])

        df = []
        for query in content['BlastOutput2']:
            results = query['report']['results']['search']
            query_info = {field:value for field, value in results.items() if (field != 'hits')}

            if (len(results['hits']) == 0):
                # df.append(query_info) # Add an empty row if there's no hits. 
                continue

            for hit in results['hits']:
                hit_info = {'len':hit['len']}
                for hsp in hit['hsps']:
                    row = query_info.copy()
                    row.update(hit_info)
                    row.update(hit['description'][0]) # Only get the description for the first entry. 
                    row.update(hsp)
                    df.append(row)
        df = pd.DataFrame(df)
        df = df[list(BLASTJsonFile.field_map.keys())].rename(columns=BLASTJsonFile.field_map) # Rename columns according to the field map. 
        
        # Need to make sure not to count the number of high-scoring pairs, just the hits. 
        n_hits = df.groupby('id').apply(lambda df : df.subject_id.nunique(dropna=True))
        n_hits.name = 'n_hits'
        n_subject_taxa = df.groupby('id').apply(lambda df : df.subject_taxonomy_id.nunique(dropna=True))
        n_subject_taxa.name = 'n_subject_taxa'
        
        df = df.set_index('id')
        df = df.merge(n_hits, left_index=True, right_index=True, how='left', validate='many_to_one') 
        df = df.merge(n_subject_taxa, left_index=True, right_index=True, how='left', validate='many_to_one') 
        self.df = df

    
    def to_df(self):
        
        df = self.df.copy()
        df = fillna(df, rules={str:'none'}, errors='ignore') # Fill in the empty sequences and subject IDs. 
        df['percent_identity'] = df['positive'] / df['alignment_length']
        df['subject_description'] = [re.sub(r'\[(.+)\]', '', description) for description in df.subject_description] # Remove the taxon name from the sequence description.
        # df['subject_description'] = df.subject_description.str.strip()
        df['subject_description'] = df['subject_description'].str.replace(r'^\s+|\s+$', '', regex=True) # Remove leading and trailing whitespace. 
        # First sort by whether or not the hit is hypothetical, and then by E-value. This means selecting the first of hits for the same query sequence
        # will first prioritize all non-hypothetical hits over hypothetical hits. If there are multiple HSPs for a single hit, the best HSP will be first. 
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