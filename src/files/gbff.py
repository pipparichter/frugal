import os 
import re 
from typing import List, Dict, Tuple 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
from src import fillna

# TODO: I should automatically detect the feature labels instead of specifying beforehand.

class GBFFFile():
    fields = ['feature', 'contig_id', 'strand', 'start', 'stop', 'partial', 'product', 'note', 'protein_id', 'seq', 'pseudo', 'locus_tag', 'inference', 'experiment']
    fields += ['evidence_type', 'evidence_category', 'evidence_details', 'evidence_source', 'translation_table', 'ribosomal_slippage', 'continuous', 'codon_start']
    dtypes = {'start':int, 'stop':int, 'strand':int, 'pseudo':bool, 'ribosomal_slippage':bool, 'continuous':bool}
    for field in fields: # Set all other fields to the string datatype.
        dtypes[field] = dtypes.get(field, str) 

    
    noncoding_features = ['tRNA', 'ncRNA', 'rRNA', 'misc_RNA', 'tmRNA']
    coding_features = ['CDS']
    nucleotide_features = ['stem_loop', 'misc_structure']
    features = noncoding_features + coding_features + nucleotide_features + ['gene', 'repeat_region', 'misc_feature', 'mobile_element', 'regulatory', 'misc_binding', 'operon', 'unsure']

    qualifier_pattern = re.compile(r'/([a-zA-Z_]+)="([^"]+)"') # Two capturing groups so that re.findall gets a list of tuples. 
    translation_table_pattern = re.compile(r'/transl_table=([0-9]+)')
    codon_start_pattern = re.compile(r'/codon_start=([0-9]+)')
    feature_pattern = r'[\s]{2,}(' + '|'.join(features) + r')[\s]{2,}'

    # The order of these coordinate patterns is important, as want to prioritize the outer match; coordinates can be of the form complement(join(..)),
    # for example, and I want to make sure I don't just match the substring. It is also important to enable multiline, as sometimes the coordinates span multiple lines.
    coordinate_pattern = re.compile('|'.join([r'(complement\(.+\))', r'(join\(.+\))', r'(order\(.+\))', r'([\<\d]+\.\.[\>\d]+)']), flags=re.MULTILINE)

    evidence_types = ['experiment']
    evidence_types += ['alignment']
    evidence_types += ['similar to sequence', 'similar to AA sequence', 'similar to DNA sequence', 'similar to RNA sequence', 'similar to RNA sequence, mRNA', 'similar to RNA sequence, EST', 'similar to RNA sequence, other RNA']
    evidence_types += ['profile', 'nucleotide motif', 'protein motif', 'ab initio prediction']
    evidence_types += ['non-experimental evidence, no additional details recorded']
    evidence_types += ['none']

    evidence_categories = ['EXISTENCE', 'COORDINATES', 'DESCRIPTION']

    @staticmethod
    def used_pgap(path:str) -> bool:
        content = ''
        with open(path, 'r') as f:
            for line in f: # Read in the file until the first set of features, and then see if PGAP is mentioned anywhere in the comment. 
                if 'FEATURES' in line:
                    break
                content += line 
        return ('PGAP' in content)

    @staticmethod
    def parse_coordinate(coordinate:str):
        '''For information on location operators, see section 3.4.2.2 of https://www.insdc.org/submitting-standards/feature-table/#3.4.2.2'''
        # In bacteria, the joins can be used in cases like programmed frameshifts (e.g. if there is ribosomal slippage)
        parsed_coordinate = list()

        strand = 1
        if re.match(r'complement\((.+)\)', coordinate) is not None:
            strand = -1 
            coordinate = re.match(r'complement\((.+)\)', coordinate).group(1) 
        
        continuous = True
        if re.match(r'join\((.+)\)', coordinate) is not None:
            continuous = False 
            coordinate = re.match(r'join\((.+)\)', coordinate).group(1)

        if re.match(r'order\((.+)\)', coordinate) is not None:
            continuous = False 
            coordinate = re.match(r'order\((.+)\)', coordinate).group(1)  

        for range_ in coordinate.split(','): # Handles the case of a potential join. 
            try: # Edge case where coordinate looks like join(876461,1..1493)
                start, stop = range_.split('..')
            except ValueError:
                start, stop = range_, range_
            partial = ('1' if ('<' in start) else '0') + ('1' if ('>' in stop) else '0')
            start, stop = int(start.replace('<', '')), int(stop.replace('>', ''))
            parsed_coordinate.append({'start':start, 'stop':stop, 'continuous':continuous, 'strand':strand, 'partial':partial})

        return parsed_coordinate

    @staticmethod
    def parse_qualifiers(qualifiers:list, delimiter:str=';'):
        # Need to account for the fact that a single entry can have multiples of the same qualifier.
        parsed_qualifiers = dict()
        for field, value in qualifiers:
            if (field not in parsed_qualifiers):
                parsed_qualifiers[field] = value
            else:
                value = value.replace(delimiter, ' ')
                # assert (delimiter not in value), f"GBFFFile.parse_qualifiers: There is already a '{delimiter}' in \"{value}\" for field {field}. Need to use a different delimiter."
                parsed_qualifiers[field] += delimiter + value
        return parsed_qualifiers

    @staticmethod
    def parse_feature(feature:str, qualifiers:str) -> dict:
        assert feature in GBFFFile.features, f'GBFFFile.parse_feature: Feature {feature} is not specified in GBFFFile.features.'

        # Extract the gene coordinates, which do not follow the typical field pattern. 
        coordinate = re.search(GBFFFile.coordinate_pattern, qualifiers).group(0)
        pseudo = ('/pseudo' in qualifiers)
        ribosomal_slippage = ('/ribosomal_slippage' in qualifiers)
        # Need a special case to extract the translation table, as it is not surrounded by double quotation marks.

        translation_table = re.search(GBFFFile.translation_table_pattern, qualifiers).group(1) if re.search(GBFFFile.translation_table_pattern, qualifiers) else 'none'
        codon_start = re.search(GBFFFile.codon_start_pattern, qualifiers).group(1) if re.search(GBFFFile.codon_start_pattern, qualifiers) else 'none'

        # Remove all newlines or any more than one consecutive whitespace character.
        # This accomodates the fact that some of the fields are multi-line. 
        qualifiers = re.sub(r'[\s]{2,}|\n', ' ', qualifiers) 
        qualifiers = re.findall(GBFFFile.qualifier_pattern, qualifiers) # Returns a list of matches. 
        qualifiers = GBFFFile.parse_qualifiers(qualifiers) # Convert the qualifiers to a dictionary. 

        parsed_feature = list()
        for parsed_coordinate in GBFFFile.parse_coordinate(coordinate):
            parsed_feature_ = dict()
            parsed_feature_['coordinate'] = coordinate
            parsed_feature_['translation_table'] = translation_table
            parsed_feature_['codon_start'] = codon_start 
            parsed_feature_['pseudo'] = pseudo
            parsed_feature_['ribosomal_slippage'] = ribosomal_slippage 
            parsed_feature_['feature'] = feature
            parsed_feature_['note'] = qualifiers.get('note', 'none')
            parsed_feature_['experiment'] = qualifiers.get('experiment', 'none')
            parsed_feature_['inference'] = qualifiers.get('inference', 'none')
            parsed_feature_['seq'] = re.sub(r'[\s]+', '', qualifiers.get('translation', 'none')) # Remove leftover whitespace in sequence. 
            parsed_feature_['product'] = qualifiers.get('product', 'none')
            parsed_feature_['protein_id'] = qualifiers.get('protein_id', 'none')
            parsed_feature_['locus_tag'] = qualifiers.get('locus_tag', 'none')
            parsed_feature_.update(parsed_coordinate)
            parsed_feature.append(parsed_feature_)

        return parsed_feature 

    @staticmethod
    def parse_contig(contig:str) -> pd.DataFrame:
        seq = re.search(r'ORIGIN(.*?)(?=(//)|$)', contig, flags=re.DOTALL).group(1)
        contig_id = contig.split()[0].strip()
        # Because I use parentheses around the features in the pattern, this is a "capturing group", and the label is contained in the output. 
        contig = re.split(GBFFFile.feature_pattern, contig, flags=re.MULTILINE)
        metadata, features = contig[0], contig[1:] # The first entry in the content is random contig metadata, like the authors. 

        if len(features) == 0: # Catches the case where the contig is not associated with any gene features. 
            return contig_id, seq, None

        features = [(features[i], features[i + 1]) for i in range(0, len(features), 2)] # Tuples of form (feature, data). 
        features = [feature for feature in features if feature[0] != 'gene'] # Remove the gene entries, which I think are kind of redundant with the products. 
        df = pd.DataFrame([parsed_feature for feature in features for parsed_feature in GBFFFile.parse_feature(*feature)])
        df['contig_id'] = contig_id

        return contig_id, seq, df 


    def __init__(self, path:str, add_evidence:bool=True):
        
        self.path, self.df, self.contigs = path, list(), dict() 

        with open(path, 'r') as f:
            content = f.read()

        # If there are multiple contigs in the file, the set of features corresponding to the contig is marked by a "contig" feature.
        # I used the lookahead match here because it is not treated as a capturing group (so I don't get LOCUS twice). 
        contigs = re.findall(r'LOCUS(.*?)(?=LOCUS|$)', content, flags=re.DOTALL) # DOTALL flag means the dot character also matches newlines.
 
        for contig in contigs:
            contig_id, contig_seq, contig_df = GBFFFile.parse_contig(contig)
            self.contigs[contig_id] = re.sub(r'[\n\s0-9]', '', contig_seq).upper() # Remove line numbers and newlines from the contig sequence
            if (contig_df is not None):
                self.df.append(contig_df)

        # It's important to reset the index after concatenating so every feature has a unique label for the subsequent evidence merging. 
        self.df = pd.concat(self.df).reset_index(drop=True)
        self.add_evidence()

    def to_df(self):
        df = self.df.copy()[GBFFFile.fields] 
        df['used_pgap'] = GBFFFile.used_pgap(self.path)
        return df

    @staticmethod
    def parse_inference(inference:str):
        parsed_inference = dict()
        
        # Remove the PubMed IDs, if present, which can interfere with the inference parsing. 
        inference = re.sub(r'\[PMID[^\]]+\]', '', inference)
        inference = re.sub(r'PMID:([0-9]+(, ){,1})+', '', inference)
        assert 'PMID' not in inference, f'GBFFFile.parse_inference: Failed to remove the PubMed ID from inference {inference}'

        # Extract the inference category, if one is present. 
        category = re.match(f"({"|".join(GBFFFile.evidence_categories)}):", inference)
        if category is not None:
            category = category.group(1)
            inference = re.sub(category + r':[\s]*', '', inference)

        try:
            type_ = inference.split(':')[0]
            inference = inference.replace(f'{type_}:', '')
            assert type_ in GBFFFile.evidence_types, f'GBFFFile.parse_inference: Type {type_} is not in the valid list of inference types.'
            
            parsed_inference['category'] = category
            parsed_inference['type'] = type_

            source = inference.split(':')[0]
            inference = inference.replace(f'{source}:', '')
            parsed_inference['source'] = source
        except:
            print(f'GBFFFile.parse_inference: Problem parsing inference {inference}.')

        parsed_inference['details'] = inference
        return parsed_inference

    @staticmethod 
    def parse_experiment(experiment:str):
        parsed_experiment = dict()
        
        category = re.match(f"({"|".join(GBFFFile.evidence_categories)}):", experiment)
        if category is not None:
            category = category.group(1)
            experiment = re.sub(category + r':[\s]*', '', experiment)

        parsed_experiment['category'] = category
        parsed_experiment['details'] = experiment
        parsed_experiment['type'] = 'experiment'
        return parsed_experiment

    @staticmethod
    def get_evidence(df:pd.DataFrame, drop_duplicates:bool=True):
        df_ = []
        for row in df.itertuples():
            # There does not seep to be evidence for every entry. 
            # assert not (pd.isnull(row.inference) and pd.isnull(row.experiment)), f'GBFFFile._get_evidence: There is no evidence for row {row.Index}.'

            if (not (row.inference == 'none')):
                for inference in row.inference.split(';'):
                    row_ = {'index':row.Index}
                    row_.update(GBFFFile.parse_inference(inference))
                    df_.append(row_)
            if (not (row.experiment == 'none')):
                for experiment in row.experiment.split(';'):
                    row_ = {'index':row.Index}
                    row_.update(GBFFFile.parse_experiment(experiment))
                    df_.append(row_)

        df_ = fillna(pd.DataFrame(df_), rules={str:'none'}, check=True)
        df_['type'] = pd.Categorical(df_['type'], categories=GBFFFile.evidence_types, ordered=True)
        df_ = df_.sort_values(by=['type'])
        df_ = df_.drop_duplicates(subset=['index'], keep='first') if drop_duplicates else df_
        df_ = df_.set_index('index')
        df_.index.name = df.index.name
        return df_

    def add_evidence(self):
        evidence_df = GBFFFile.get_evidence(self.df, drop_duplicates=True)
        evidence_df.columns = ['evidence_' + col for col in evidence_df.columns]
        df = self.df.merge(evidence_df, left_index=True, right_index=True, how='left')
        df[evidence_df.columns] = fillna(df[evidence_df.columns].copy(), rules={str:'none'}, check=True) # Fill the things which became NaN in the merge.
        self.df = df




# Qualifier       /inference=
# Definition      a structured description of non-experimental evidence that supports
#                 the feature identification or assignment.

# Value format    "[CATEGORY:]TYPE[ (same species)][:EVIDENCE_BASIS]"
  
#                 where CATEGORY is one of the following:
#                 "COORDINATES" support for the annotated coordinates
#                 "DESCRIPTION" support for a broad concept of function such as that
#                 based on phenotype, genetic approach, biochemical function, pathway
#                 information, etc.
#                 "EXISTENCE" support for the known or inferred existence of the product
  
#                 where TYPE is one of the following:
#                 "non-experimental evidence, no additional details recorded"
#                    "similar to sequence"
#                       "similar to AA sequence"
#                       "similar to DNA sequence"
#                       "similar to RNA sequence"
#                       "similar to RNA sequence, mRNA"
#                       "similar to RNA sequence, EST"
#                       "similar to RNA sequence, other RNA"
#                    "profile"
#                       "nucleotide motif"
#                       "protein motif"
#                       "ab initio prediction"
#                    "alignment"

type_order = ['similar to ']
type_order = ["similar to sequence", "similar to AA sequence", "similar to DNA sequence", "similar to RNA sequence", "similar to RNA sequence, mRNA", "similar to RNA sequence, EST", "similar to RNA sequence, other RNA"]
  
#                 where the optional text "(same species)" is included when the
#                 inference comes from the same species as the entry.
  
#                 where the optional "EVIDENCE_BASIS" is either a reference to a
#                 database entry (including accession and version) or an algorithm
#                 (including version) , eg 'INSD:AACN010222672.1', 'InterPro:IPR001900',
#                 'ProDom:PD000600', 'Genscan:2.0', etc. and is structured 
#                 "[ALGORITHM][:EVIDENCE_DBREF[,EVIDENCE_DBREF]*[,...]]"
# Example         /inference="COORDINATES:profile:tRNAscan:2.1"
#                 /inference="similar to DNA sequence:INSD:AY411252.1"
#                 /inference="similar to RNA sequence, mRNA:RefSeq:NM_000041.2"
#                 /inference="similar to DNA sequence (same
#                 species):INSD:AACN010222672.1"
#                 /inference="protein motif:InterPro:IPR001900"
#                 /inference="ab initio prediction:Genscan:2.0"
#                 /inference="alignment:Splign:1.0"
#                 /inference="alignment:Splign:1.26p:RefSeq:NM_000041.2,INSD:BC003557.1"

# Comment         /inference="non-experimental evidence, no additional details 
#                 recorded" was used to replace instances of 
#                 /evidence=NOT_EXPERIMENTAL in December 2005; any database ID can be
#                 used in /inference= qualifier; recommentations for choice of resource 
#                 acronym for[EVIDENCE_BASIS] are provided in the /inference qualifier
#                 vocabulary recommendation document (https://www.insdc.org/submitting-standards/inference-qualifiers/); 

# Qualifier       /experiment=
# Definition      a brief description of the nature of the experimental 
#                 evidence that supports the feature identification or assignment.
# Value format    "[CATEGORY:]text"
#                 where CATEGORY is one of the following:
#                 "COORDINATES" support for the annotated coordinates
#                 "DESCRIPTION" support for a broad concept of function such as that
#                 based on phenotype, genetic approach, biochemical function, pathway
#                 information, etc.
#                 "EXISTENCE" support for the known or inferred existence of the product
#                 where text is free text (see examples)
# Example         /experiment="5' RACE"
#                 /experiment="Northern blot [DOI: 12.3456/FT.789.1.234-567.2010]"
#                 /experiment="heterologous expression system of Xenopus laevis
#                 oocytes [PMID: 12345678, 10101010, 987654]"
#                 /experiment="COORDINATES: 5' and 3' RACE"
# Comment         detailed experimental details should not be included, and would
#                 normally be found in the cited publications; PMID, DOI and any 
#                 experimental database ID is allowed to be used in /experiment
#                 qualifier; Please also visit: https://www.insdc.org/submitting-standards/recommendations-vocabulary-insdc-experiment-qualifiers/; 
#                 value "experimental evidence, no additional details recorded"
#                 was used to  replace instances of /evidence=EXPERIMENTAL in
#                 December 2005