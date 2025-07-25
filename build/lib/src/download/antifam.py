import os 
import pandas as pd 
import subprocess
from tqdm import tqdm 
import shutil
import numpy as np 
from src.files import FASTAFile
import io 
from src import fillna
import requests 
import json

class AntiFam():

    def __init__(self):
        pass 

    def get_antifam_ids(self, path:str='../data/antifam_ids.json'):
        antifam_ids = []
        result = json.loads(requests.get('https://www.ebi.ac.uk/interpro/api/entry/antifam/').text)
        pbar = tqdm(total=result['count'], desc='AntiFam.get_antifams')
        while True: # Only returns 20 hits at a time, so need to paginate using the 'next' field. 
            antifam_ids += [{'id':entry['metadata']['accession'], 'description':entry['metadata']['name']} for entry in result['results']]
            pbar.update(len(result['results']))
            if result['next'] is None:
                break
            result = json.loads(requests.get(result['next']).text)
        with open(path, 'w') as f:
            json.dump(antifam_ids, f)
        print(f'AntiFam.get_antifams: IDs for {len(antifam_ids)} AntiFam families written to {path}')

    @staticmethod
    def _get_protein_info(entry:dict, antifam_id:str='none') -> dict:
        '''Extract information in a JSON entry for a protein.'''
        metadata = entry['metadata']
        info = dict()
        info['id'] = metadata['accession']
        info['antifam_id'] = antifam_id
        info['product'] = metadata['name']
        info['ncbi_taxonomy_id'] = metadata['source_organism']['taxId']
        # info['organism'] = metadata['source_organism']['fullName']
        return info
    
    def get_proteins(self, antifam_ids:list, path:str=None) -> pd.DataFrame:
        '''Obtain the UniProt IDs for the protein entries associated with each AntiFam in the InterPro database.'''
        df = []
        for id_ in antifam_ids:
            url = 'https://www.ebi.ac.uk/interpro/api/protein/unreviewed/entry/antifam/{antifam}?'.format(antifam=id_)
            result = json.loads(requests.get(url).text)
            pbar = tqdm(total=result['count'], desc='Downloading sequences for AntiFam family {antifam}.'.format(antifam=id_))
            while True:
                pbar.update(len(result['results']))
                df += [AntiFam._get_protein_info(entry, antifam_id=id_) for entry in result['results']]
                if result['next'] is None:
                    break
                result = json.loads(requests.get(result['next']).text)
            pbar.close()

        df = pd.DataFrame(df).set_index('id')
        df.to_csv(path)
        print(f'AntiFam.get_proteins: AntiFam protein data for {len(df)} sequences written to {path}')
        return df 
