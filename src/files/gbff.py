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
    fields = ['feature', 'contig_id', 'strand', 'start', 'stop', 'partial', 'product', 'note', 'protein_id', 'seq', 'pseudo', 'locus_tag', 'inference', 'experiment']
    fields += ['evidence_type', 'evidence_category', 'evidence_details', 'evidence_source']
    dtypes = {'start':int, 'stop':int, 'strand':int, 'pseudo':bool}
    features = ['gene', 'CDS', 'tRNA', 'ncRNA', 'rRNA', 'misc_RNA','repeat_region', 'misc_feature', 'mobile_element']

    qualifier_pattern = re.compile(r'/([a-zA-Z_]+)="([^"]+)"') # Two capturing groups so that re.finall gets a list of tuples. 
    coordinate_pattern = re.compile(r'complement\([\<\d]+\.\.[\>\d]+\)|[\<\d]+\.\.[\>\d]+')
    feature_pattern = r'[\s]{2,}(' + '|'.join(features) + r')[\s]{2,}'

    evidence_types = ["experiment"]
    evidence_types += ["alignment"]
    evidence_types += ["similar to sequence", "similar to AA sequence", "similar to DNA sequence", "similar to RNA sequence", "similar to RNA sequence, mRNA", "similar to RNA sequence, EST", "similar to RNA sequence, other RNA"]
    evidence_types += ["profile", "nucleotide motif", "protein motif", "ab initio prediction"]
    evidence_types += ["non-experimental evidence, no additional details recorded"]

    evidence_categories = ['EXISTENCE', 'COORDINATES', 'DESCRIPTION']

    @staticmethod
    def used_pgap(path:str) -> bool:
        content = ''
        with open(path, 'r') as f:
            for line in f:
                # Read in the file until the first set of features, and then see if PGAP
                # is mentioned anywhere in the comment. 
                if 'FEATURES' in line:
                    break
                content += line 
        return ('PGAP' in content)

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
    def parse_feature(feature:str, feature_data:str) -> dict:
        assert feature in GBFFFile.features, f'GBFFFile.parse_feature: Feature {feature} is not specified in GBFFFile.features.'

        # Extract the gene coordinates, which do not follow the typical field pattern. 
        coordinate = re.search(GBFFFile.coordinate_pattern, feature_data).group(0)
        pseudo = ('/pseudo' in feature_data)
        feature_data = re.sub(GBFFFile.coordinate_pattern, '', feature_data)

        # Remove all newlines or any more than one consecutive whitespace character.
        # This accomodates the fact that some of the fields are multi-line. 
        feature_data = re.sub(r'[\s]{2,}|\n', ' ', feature_data) 
        qualifiers = re.findall(GBFFFile.qualifier_pattern, feature_data) # Returns a list of matches. 
        qualifiers = GBFFFile.parse_qualifiers(qualifiers)

        parsed_feature_data = dict()
        parsed_feature_data['coordinate'] = coordinate
        parsed_feature_data['pseudo'] = pseudo
        parsed_feature_data['feature'] = feature
        parsed_feature_data['note'] = qualifiers.get('note', '')
        parsed_feature_data['experiment'] = qualifiers.get('experiment', None)
        parsed_feature_data['inference'] = qualifiers.get('inference', None)
        parsed_feature_data['seq'] = qualifiers.get('translation', None)
        parsed_feature_data['product'] = qualifiers.get('product', None)
        parsed_feature_data['protein_id'] = qualifiers.get('protein_id', None)
        parsed_feature_data['locus_tag'] = qualifiers.get('locus_tag', None)
        parsed_feature_data.update(GBFFFile.parse_coordinate(coordinate))

        return parsed_feature_data 

    @staticmethod
    def parse_contig(contig:str) -> pd.DataFrame:
        seq = re.search(r'ORIGIN(.*?)(?=//)', contig, flags=re.DOTALL).group(1)
        contig_id = contig.split()[0].strip()
        # Because I use parentheses around the features in the pattern, this is a "capturing group", and the label is contained in the output. 
        contig = re.split(GBFFFile.feature_pattern, contig, flags=re.MULTILINE)
        metadata, features = contig[0], contig[1:] # The first entry in the content is random contig metadata, like the authors. 

        if len(features) == 0: # Catches the case where the contig is not associated with any gene features. 
            return contig_id, seq, None

        features = [(features[i], features[i + 1]) for i in range(0, len(features), 2)] # Tuples of form (feature, data). 
        features = [feature for feature in features if feature[0] != 'gene'] # Remove the gene entries, which I think are kind of redundant with the products. 
        df = pd.DataFrame([GBFFFile.parse_feature(*feature) for feature in features])
        df = df.rename(columns={'translation':'seq'}) 
        df['contig_id'] = contig_id

        return contig_id, seq, df 

    @staticmethod
    def clean_nt_seq(nt_seq:str):
        nt_seq = re.sub(r'[\n\s0-9]', '', nt_seq)
        nt_seq = nt_seq.upper()
        return nt_seq

    def __init__(self, path:str, add_evidence:bool=True):
        
        self.path, self.df, self.contigs = path, list(), dict() 

        with open(path, 'r') as f:
            content = f.read()

        # If there are multiple contigs in the file, the set of features corresponding to the contig is marked by a "contig" feature.
        # I used the lookahead match here because it is not treated as a capturing group (so I don't get LOCUS twice). 
        contigs = re.findall(r'LOCUS(.*?)(?=LOCUS|$)', content, flags=re.DOTALL) # DOTALL flag means the dot character also matches newlines.
 
        for contig in contigs:
            contig_id, contig_seq, contig_df = GBFFFile.parse_contig(contig)
            self.contigs[contig_id] = GBFFFile.clean_nt_seq(contig_seq)
            if (contig_df is not None):
                self.df.append(contig_df)
        # It's important to reset the index after concatenating so every feature has a unique label for the subsequent evidence merging. 
        self.df = pd.concat(self.df).reset_index(drop=True)
        self.add_evidence()


    def check(self):
        for field, dtype in GBFFFile.dtypes.items():
            assert self.df[field].dtype == dtype, f'GBFFFile.check: Data type is incorrect for field {col}.'

    def to_df(self):
        self.check()
        df = self.df.copy()[GBFFFile.fields] 
        df = df[~df.locus_tag.isnull()]
        df['used_pgap'] = GBFFFile.used_pgap(self.path)
        return df

    @staticmethod
    def parse_inference(inference:str):
        parsed_inference, input_inference = dict(), inference
        
        inference = re.sub(r'\[PMID[^\]]+\]', '', inference)
        inference = re.sub(r'PMID:([0-9]+(, ){,1})+', '', inference)
        assert 'PMID' not in inference, f'GBFFFile.parse_inference: Failed to remove the PubMed ID from inference {inference}'

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
            pass
            # print(f'GBFFFile.parse_inference: Problem parsing inference {input_inference}.')

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
    def _get_evidence(df:pd.DataFrame, best_evidence_only:bool=True, errors:str='raise'):
        df_ = []
        for row in df.itertuples():
            # There should be at least *some* evidence for every entry. 
            if errors == 'raise':
                assert not (pd.isnull(row.inference) and pd.isnull(row.experiment)), f'GBFFFile._get_evidence: There is no evidence for row {row.Index}.'

            if (not pd.isnull(row.inference)):
                for inference in row.inference.split(';'):
                    row_ = {'index':row.Index}
                    row_.update(GBFFFile.parse_inference(inference))
                    df_.append(row_)
            if (not pd.isnull(row.experiment)):
                for experiment in row.experiment.split(';'):
                    row_ = {'index':row.Index}
                    row_.update(GBFFFile.parse_inference(experiment))
                    df_.append(row_)
        df_ = pd.DataFrame(df_)
        df_['type'] = pd.Categorical(df_['type'], categories=GBFFFile.evidence_types, ordered=True)
        df_['category'] = pd.Categorical(df_['category'], categories=GBFFFile.evidence_categories, ordered=True)

        if best_evidence_only:
            df_ = df_.sort_values(by=['type', 'category']).drop_duplicates(subset=['index'], keep='first')
        df_ = df_.set_index('index')
        df_.index.name = df.index.name
        return df_

    def add_evidence(self):
        evidence_df = GBFFFile._get_evidence(self.df, best_evidence_only=True, errors='ignore')
        evidence_df.columns = ['evidence_' + col for col in evidence_df.columns]
        assert self.df.index.nunique() == len(self.df)
        df = self.df.merge(evidence_df, left_index=True, right_index=True, how='left')

        assert len(df) == len(self.df), f'GBFFFile.add_evidence: Error while merging evidence DataFrame. Expected merged DataFrame of size {len(self.df)}, but got {len(df)}.'
        # assert len(evidence_df) == len(self.df), 'GBFFFile.add_evidence: There should be exactly one best evidence qualifier per feature.'
        
        # Looking at the source files, it seems that no evidence qualifiers is not a bad thing, necessarily. The M. tuberculosis genome, which is manually curated,
        # is missing many of these. 
        # if df.evidence_details.isnull().sum() > 0: 
        #     print(f'\nGBFFFile.add_evidence: There is no provided evidence for {df.evidence_details.isnull().sum()} out of {len(self.df)} features.')
        
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