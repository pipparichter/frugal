import os 
import re 
from typing import List, Dict, Tuple 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 

# TODO: I should automatically detect the features instead of specifying beforehand.
# TODO: Make sure that dropping things with null locus tags is not causing issues. I did it because repeat regions are not assigned 
#   locus tags, and trying to merge during search is annoying. 

class GBFFFile():
    fields = ['feature', 'contig_id', 'strand', 'start', 'stop', 'partial', 'product', 'frameshifted', 'incomplete', 'internal_stop', 'protein_id', 'seq', 'pseudo', 'locus_tag']
    dtypes = {'start':int, 'stop':int, 'strand':int}
    features = ['gene', 'CDS', 'tRNA', 'ncRNA', 'rRNA', 'misc_RNA','repeat_region', 'misc_feature', 'mobile_element']

    field_pattern = re.compile(r'/([a-zA-Z_]+)="([^"]+)"')
    coordinate_pattern = re.compile(r'complement\([\<\d]+\.\.[\>\d]+\)|[\<\d]+\.\.[\>\d]+')
    feature_pattern = r'[\s]{2,}(' + '|'.join(features) + r')[\s]{2,}'

    @staticmethod
    def parse_coordinate(coordinate:str):
        '''Parse a string indicating gene boundaries. These strings contain information about the start codon location, 
        stop codon location, and strand.'''
        parsed_coordinate = dict()
        parsed_coordinate['strand'] = -1 if ('complement' in coordinate) else 1
        # NOTE: Details about coordinate format: https://www.ncbi.nlm.nih.gov/genbank/samplerecord/
        start, stop = re.findall(r'[\>\<0-9]+', coordinate)
        partial = ('1' if ('<' in start) else '0') + ('1' if ('>' in stop) else '0')
        start = int(start.replace('<', ''))
        stop = int(stop.replace('>', ''))

        parsed_coordinate['start'] = start
        parsed_coordinate['stop'] = stop
        parsed_coordinate['partial'] = partial

        return parsed_coordinate

    @staticmethod
    def parse_note(note:str):
        parsed_note = dict()
        parsed_note['frameshifted'] = ('frameshifted' in note)
        parsed_note['internal_stop'] = ('internal stop' in note)
        parsed_note['incomplete'] = ('incomplete' in note)
        return parsed_note 


    # NOTE: PGAP annotations can include pogrammed frameshift sequences (is this where the duplicates come in?)
    @staticmethod
    def parse_entry(feature:str, entry:str) -> dict:

        # Extract the gene coordinates, which do not follow the typical field pattern. 
        coordinate = re.search(GBFFFile.coordinate_pattern, entry).group(0)
        pseudo = ('/pseudo' in entry)
        entry = re.sub(GBFFFile.coordinate_pattern, '', entry)

        entry = re.sub(r'[\s]{2,}|\n', '', entry) # Remove all newlines or any more than one consecutive whitespace character.

        entry = re.findall(GBFFFile.field_pattern, entry) # Returns a list of matches. 
        parsed_entry = {'feature':feature}
        # Need to account for the fact that a single entry can have multiple GO functions, processes, components, etc.
        for field, value in entry:
            if (field not in parsed_entry):
                parsed_entry[field] = value
            else:
                parsed_entry[field] += ', ' + value
        parsed_entry['coordinate'] = coordinate
        parsed_entry['pseudo'] = pseudo
        parsed_entry.update(GBFFFile.parse_coordinate(coordinate))
        if 'note' in parsed_entry:
            parsed_entry.update(GBFFFile.parse_note(parsed_entry['note']))
       
        return parsed_entry 

    @staticmethod
    def parse_contig(contig:str) -> pd.DataFrame:
        seq = re.search(r'ORIGIN(.*?)(?=//)', contig, flags=re.DOTALL).group(1)
        contig_id = contig.split()[0].strip()
        # Because I use parentheses around the features in the pattern, this is a "capturing group", and the label is contained in the output. 
        contig = re.split(GBFFFile.feature_pattern, contig, flags=re.MULTILINE)
        metadata, contig = contig[0], contig[1:] # The first entry in the content is random contig metadata, like the authors. 
        # id_ = re.search(r'VERSION[\s]+([^\n]+)', metadata).group(1) # Extract the contig ID from the metadata. 

        if len(contig) == 0: # Catches the case where the contig is not associated with any gene features. 
            return contig_id, seq, None

        entries = [(contig[i], contig[i + 1]) for i in range(0, len(contig), 2)] # Tuples of form (feature, data). 
        assert np.all(np.isin([entry[0] for entry in entries], GBFFFile.features)), 'GBFFFile.__init__: Something went wrong while parsing the file.'
        entries = [entry for entry in entries if entry[0] != 'gene'] # Remove the gene entries, which I think are kind of redundant with the products. 

        df = []
        for entry in entries:
            df.append(GBFFFile.parse_entry(*entry))
        df = pd.DataFrame(df)
        df = df.rename(columns={col:col.lower() for col in df.columns}) # Make sure all column names are lower-case.
        df = df.rename(columns={'translation':'seq'}) 
        df['contig_id'] = contig_id

        return contig_id, seq, df 

    @staticmethod
    def clean_nt_seq(nt_seq:str):
        nt_seq = re.sub(r'[\n\s0-9]', '', nt_seq)
        nt_seq = nt_seq.upper()
        return nt_seq


    def __init__(self, path:str):
        
        with open(path, 'r') as f:
            content = f.read()

        # If there are multiple contigs in the file, the set of features corresponding to the contig is marked by a "contig" feature.
        # NOTE: I used the lookahead match here because it is not treated as a capturing group (so I don't get LOCUS twice). 
        contigs = re.findall(r'LOCUS(.*?)(?=LOCUS|$)', content, flags=re.DOTALL) # DOTALL flag means the dot character also matches newlines.

        self.df = list()
        self.contigs = dict() # Store the nucleotide sequences for the contigs. 

        for contig in contigs:
            contig_id, contig_seq, contig_df = GBFFFile.parse_contig(contig)
            self.contigs[contig_id] = GBFFFile.clean_nt_seq(contig_seq)
            if (contig_df is not None):
                self.df.append(contig_df)

        self.df = pd.concat(self.df)

    def to_df(self):
        df = self.df.copy()[GBFFFile.fields] 
        df = df.astype(GBFFFile.dtypes)
        df = df[~df.locus_tag.isnull()]
        return df
