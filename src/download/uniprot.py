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
import re 

class UniProt():

    xml_id_pattern = re.compile('<accession>([^<]+)</accession>')
    fasta_id_pattern = re.compile(r'>([^\s]+)')
    url = 'https://rest.uniprot.org/uniprotkb/stream?format={format_}&query=' # '%28%28accession%3AA0A0S2IWG5%29+OR+%28accession%3AA0A1T1DM31%29%29'
    
    def __init__(self):
        pass 

    @staticmethod
    def get_proteins(protein_ids:list, format_:str='xml', path:str=None, chunk_size:int=10):

        existing_protein_ids = UniProt._get_existing_ids(path)
        protein_ids = [id_ for id_ in protein_ids if (id_ not in existing_protein_ids)]
        
        print(f'UniProt.get_proteins: {len(existing_protein_ids)} sequences already present in {path}. Downloading {len(protein_ids)} new sequences.')
        
        f = open(path, 'a')

        n_chunks = len(protein_ids) // chunk_size + 1
        chunks = [protein_ids[i:i + chunk_size] for i in range(0, n_chunks * chunk_size, chunk_size)]
        pbar = tqdm(desc='UniProt.get_proteins', total=len(protein_ids))

        for chunk in chunks:
            url = UniProt.url + '+OR+'.join([f'%28accession%3A{id_}%29' for id_ in chunk])
            url = url.format(format_=format_)
            # text = requests.get(f'https://rest.uniprot.org/uniprotkb/{id_}.{format_}').text 
            text = requests.get(url).text 
            if '<errorInfo>' in text:
                print(text)
                raise Exception(f'UniProt.get_proteins: Failure on URL {url}')
            f.write(text + '\n')
            # except:
            #     print(f'UniProt.get_proteins: Failed to download {len(chunk)} sequences.')
            #     failure_protein_ids += chunk
            pbar.update(len(chunk))
        pbar.close()
        f.close()
    
    def _get_existing_ids(path:str, format_:str='xml'):
        ids = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()
            id_pattern = UniProt.xml_id_pattern if (format_ == 'xml') else UniProt.fasta_id_pattern
            ids = re.findall(id_pattern, content)
        return ids

