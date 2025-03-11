import pandas as pd 
import numpy as np 

# TODO: What does it mean if an InterPro hit has a null E-value? I think it might be related to the type of hit, e.g. MobiDBLite hits (consensus disorder predictions)
#   do not have associated E-values, but I should confirm. 

class InterProScanFile():

    fields = ['id', 'md5', 'length', 'analysis', 'signature', 'signature_description', 'start', 'stop', 'e_value', 'status', 'date', 'annotation', 'description', 'go_annotation', 'pathways_annotation']
    minimal_fields = ['analysis', 'signature_description', 'e_value', 'annotation', 'description']
    
    dtypes = {'e_value':float, 'status':bool, 'start':int, 'stop':int, 'length':int}
    analyses = ['Pfam', 'NCBIfam' 'TIGRFAM', 'Funfam', 'SFLD', 'PANTHER', 'Hamap', 'ProSiteProfiles', 'SMART', 'CDD', 'PRINTS', 'PIRSIR', 'ProSitePatterns', 'Antifam', 'MobiDBLite', 'Gene3D']

    prefix = 'interpro'
    
    def __init__(self, path:str, format_:str='tsv'):

        self.df = pd.read_csv(path, delimiter='\t', names=InterProScanFile.fields)
        self.df = self.df.set_index('id')
    
    def to_df(self, max_e_value:float=None, drop_duplicates:bool=False, use_minimal_fields:bool=True, add_prefix:bool=True):
        
        df = self.df.drop(columns=['go_annotation', 'pathways_annotation', 'md5']).copy()
        df['status'] = [True if (status == 'T') else False for status in df.status]
        df['e_value'] = df.e_value.replace({'-':np.nan})
        df = df.replace({'-':'none'}) 
        df = df.astype(InterProScanFile.dtypes)
        df = df.sort_values('e_value', ascending=True) 

        if max_e_value is not None:
            df = df[df.e_value < max_e_value].copy()
        if drop_duplicates:
            # Because E-values have been sorted, this will keep the hit with the best E-value.
            # NaN is treated as the highest element when sorting, so this will prioritize non-NaN hits.
            df = df[~df.index.duplicated(keep='first')]
        if use_minimal_fields:
            df = df[InterProScanFile.minimal_fields].copy()
        if add_prefix:
            df = df.rename(columns={col:f'{InterProScanFile.prefix}_{col}' for col in df.columns})

        return df

    def __len__(self):
        return len(self.df)

# TIGRFAM (XX.X) : TIGRFAMs are protein families based on hidden Markov models (HMMs).
# SFLD (X) : SFLD is a database of protein families based on hidden Markov models (HMMs).
# SUPERFAMILY (X.XX) : SUPERFAMILY is a database of structural and functional annotations for all proteins and genomes.
# PANTHER (XX.X) : The PANTHER (Protein ANalysis THrough Evolutionary Relationships) Classification System is a unique resource that classifies genes by their functions, using published scientific experimental evidence and evolutionary relationships to predict function even in the absence of direct experimental evidence.
# Gene3D (X.X.X) : Structural assignment for whole genes and genomes using the CATH domain structure database.
# Hamap (XXXX_XX) : High-quality Automated and Manual Annotation of Microbial Proteomes.
# ProSiteProfiles (XXX_XX) : PROSITE consists of documentation entries describing protein domains, families and functional sites as well as associated patterns and profiles to identify them.
# Coils (X.X.X) : Prediction of coiled coil regions in proteins.
# SMART (X.X) : SMART allows the identification and analysis of domain architectures based on hidden Markov models (HMMs).
# CDD (X.XX) : CDD predicts protein domains and families based on a collection of well-annotated multiple sequence alignment models.
# PRINTS (XX.X) : A compendium of protein fingerprints - a fingerprint is a group of conserved motifs used to characterise a protein family.
# PIRSR (XXXX_XX) : PIRSR is a database of protein families based on hidden Markov models (HMMs) and Site Rules.
# ProSitePatterns (XXXX_XX) : PROSITE consists of documentation entries describing protein domains, families and functional sites as well as associated patterns and profiles to identify them.
# AntiFam (X.X) : AntiFam is a resource of profile-HMMs designed to identify spurious protein predictions.
# Pfam (XX.X) : A large collection of protein families, each represented by multiple sequence alignments and hidden Markov models (HMMs).
# MobiDBLite (X.X) : Prediction of intrinsically disordered regions in proteins.
# PIRSF (X.XX) : The PIRSF concept is used as a guiding principle to provide comprehensive and non-overlapping clustering of UniProtKB sequences into a hierarchical order to reflect their evolutionary relationships.
