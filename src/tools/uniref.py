import os 
import pandas as pd 
import subprocess
from tqdm import tqdm 
from typing import List
import numpy as np 
import warnings
import glob
from src.files import FASTAFile
import requests 
import re 

# TODO: Figure out why wget was not working, e.g. wget --content-disposition=off "https://rest.uniprot.org/uniref/stream?format=fasta&query=%28UniRef50_A0A5C4Y7B5+UniRef90_UPI0004951B86%29" -o test.faa

class UniRef():
    url = 'https://rest.uniprot.org/uniref/stream?format=fasta&query=%28{protein_ids}%29'
    def __init__(self):
        pass

    def run(self, protein_ids:list=None, path:str=None, chunk_size:int=20):
        n_chunks = len(protein_ids) // chunk_size + 1
        protein_ids = ['+OR+'.join(protein_ids[i:i + chunk_size]) for i in range(0, n_chunks * chunk_size, chunk_size)]
        
        df = list()
        for protein_ids_ in protein_ids:
            url = UniRef.url.format(protein_ids=protein_ids)
            cmd = f'wget "{url}"'

            output = subprocess.run(cmd, shell=True, check=True, capture_output=True)
            output = output.stderr.decode('utf-8')
            src_path = re.search(r"Saving to: ([^\n]+)", output).group(1)[1:-1]

            df.append(FASTAFile(src_path).to_df(prodigal_output=False))
            os.remove(src_path)
        df = pd.concat(df)

        if path is not None:
            print(f'NCBIDatasets._get_proteins: Proteins saved to {path}')
            FASTAFile(df=df).write(path)
        return df 
    
    def cleanup(self):
        pass 
        