import os 
import re 
from typing import List, Dict, Tuple, NoReturn
from tqdm import tqdm 
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd 


def parser_prodigal(description:str):
    pattern = r'# ([\d]+) # ([\d]+) # ([-1]+) # ID=([^;]+);partial=([^;]+);start_type=([^;]+);rbs_motif=([^;]+);rbs_spacer=([^;]+);gc_cont=([\.\w]+)'
    columns = ['start', 'stop', 'strand', 'ID', 'partial', 'start_type', 'rbs_motif', 'rbs_spacer', 'gc_content']
    match = re.search(pattern, description)
    parsed_header = {col:match.group(i + 1) for i, col in enumerate(columns)}
    parsed_header['rbs_motif'] = 'none' if (parsed_header['rbs_motif'] == 'None') else parsed_header['rbs_motif']
    return parsed_header


def parser_default(description:str):
    return {'description':description}


class FASTAFile():
    prodigal_dtypes = {'start':int, 'stop':int, 'strand':int, 'gc_content':float, 'rbs_motif':str, 'rbs_spacer':str}

    def __init__(self, path:str=None, df:pd.DataFrame=None):
        '''Initialize a FASTAFile object.'''

        if (path is not None):
            f = open(path, 'r')
            self.seqs, self.ids, self.descriptions = [], [], []
            for record in SeqIO.parse(path, 'fasta'):
                self.ids.append(record.id)
                self.descriptions.append(record.description.replace(record.id, '').strip())
                self.seqs.append(str(record.seq))
            f.close()
            
        if (df is not None):
            self.seqs = df.seq.values
            self.ids = df.index.values 
            self.descriptions = df.description.values if ('description' in df.columns) else [''] * len(self.ids)
        
        self.seqs = [seq.replace(r'*', '') for seq in self.seqs] # Remove the terminal * character if present.

    def __len__(self):
        return len(self.seqs)
            
    def to_df(self, prodigal_output:bool=True) -> pd.DataFrame:

        parser = parser_prodigal if prodigal_output else parser_default

        df = []
        for id_, seq, description in zip(self.ids, self.seqs, self.descriptions):
            row = parser(description)
            row['id'] = id_
            row['seq'] = seq
            df.append(row)
        df = pd.DataFrame(df).set_index('id')
        if prodigal_output:
            df['contig_id'] = [id_.split('.')[0] for id_ in df.index]
            df = df.astype(FASTAFile.prodigal_dtypes)
        return df

    def write(self, path:str, mode:str='w') -> NoReturn:
        f = open(path, mode=mode)
        records = []
        for id_, seq, description in zip(self.ids, self.seqs, self.descriptions):
            record = SeqRecord(Seq(seq), id=id_, description=description)
            records.append(record)
        SeqIO.write(records, f, 'fasta')
        f.close()