from src.files.gbff import GBFFFile
import pandas as pd 
import numpy as np
from Bio.Seq import Seq
from tqdm import tqdm 


class ReferenceGenome():

    def __init__(self, path:str, genome_id:str=None):

        file = GBFFFile(path) 

        self.genome_id = genome_id
        self.df = file.to_df()
        self.contigs = file.contigs

        self.labels = dict()
        self.label_info = dict()

        self._add_lengths()
        self._add_start_stop_codons()

    def __str__(self):
        return self.genome_id
    
    def __len__(self):
        return len(self.df)

    def _add_lengths(self):
        # Can't just use seq.apply(len) because forcing the sequences to be strings causes null sequences (e.g., in the case of non-CDS features) to be 'nan'.'''
        # This also gets lengths for pseudogenes. 
        lengths = list()
        for row in self.df.itertuples():
            if (row.feature == 'CDS'):
                lengths.append((row.stop - row.start) // 3) # The start and stop indices are in terms of nucleotides. 
            else:
                lengths.append(None)
        self.df['length'] = lengths 

    def _add_start_stop_codons(self):
        self.df = self.add_start_stop_codons(self.df)

    def _merge(self, df:pd.DataFrame, hits_df:pd.DataFrame):
        n = len(df)

        df_no_hits = df[hits_df.n_valid_hits == 0].copy()
        print(f'ReferenceGenome._merge: {len(df_no_hits)} entries in the query DataFrame had no valid hits in the reference.')
        
        hits_df = hits_df[hits_df.n_valid_hits > 0].rename(columns={'locus_tag':'ref_locus_tag', 'feature':'ref_feature'})
        ref_df = self.df.copy().rename(columns={col:'ref_' + col for col in self.df.columns})
        # The locus tags are not unique, so we can't use it to merge, also need to use the features... 
        hits_df = hits_df.merge(ref_df, on=['ref_locus_tag', 'ref_feature'], how='left').set_index(hits_df.index)

        df = hits_df.merge(df, left_index=True, right_index=True, how='left')        
        df.index.name = 'id' # Restore the index name, which is lost in the merging. 
        df = pd.concat([df, df_no_hits], ignore_index=False)
        
        assert (len(df) == n), f'ReferenceGenome._merge: There was a problem merging the query DataFrame with the search results.'
        return df

    def _get_hit(self, query):

        ref_df = self.df.copy() 
        ref_df = ref_df[ref_df.contig_id == query.contig_id]

        # Filter out everything which definitely has no overlap.
        ref_df = ref_df[~(ref_df.start > query.stop) & ~(ref_df.stop < query.start)]

        # Case where there are no detected genes on a contig in the GBFF file, but Prodigal found one.
        if len(ref_df) == 0: 
            return {'n_hits':0, 'n_valid_hits':0, 'feature':None, 'locus_tag':None}

        ref_df['valid_hit'] = ((ref_df.stop == query.stop) | (ref_df.start == query.start)) & (ref_df.strand == query.strand)
        n_hits, n_valid_hits = len(ref_df), ref_df.valid_hit.sum().item() 

        if n_valid_hits > 0:
            ref_df = ref_df[ref_df.valid_hit] # Filter by the valid hits. 
            # There are some cases with more than one valid hit, in which case we want to grab the hit with the biggest overlap. 
            # It might be interesting to look into these cases a bit more, but they are very rare. 
            ref_df['overlap'] = ref_df.apply(lambda row : len(np.intersect1d(np.arange(row.start, row.stop), np.arange(query.start, query.stop))), axis=1)
            ref_df = ref_df.sort_values(by='overlap', ascending=False).iloc[[0]]

        if n_valid_hits == 0:
            return {'n_hits':n_hits, 'n_valid_hits':0, 'feature':None, 'locus_tag':None} 

        feature = ref_df['feature'].iloc[0]
        locus_tag = ref_df['locus_tag'].iloc[0]
        return {'n_hits':n_hits, 'n_valid_hits':n_valid_hits, 'feature':feature, 'locus_tag':locus_tag}  


    def search(self, df:pd.DataFrame, add_start_stop_codons:bool=True):
        hits_df = list()
        for query in tqdm(list(df.itertuples()), desc='ReferenceGenome.search'):
            hit = self._get_hit(query) # Get the hit with the biggest overlap, with a preference for "valid" hits. 
            hit['id'] = query.Index
            hits_df.append(hit)
        hits_df = pd.DataFrame(hits_df).set_index('id')
        df = self._merge(df, hits_df)

        if add_start_stop_codons:
            df = self.add_start_stop_codons(df)
        return df


    def get_nt_seq(self, start:int=None, stop:int=None, strand:int=None, contig_id:int=None, error:str='ignore'):
        nt_seq = self.contigs[contig_id] 
        # Pretty sure the stop position is non-inclusive, so need to shift it over.
        nt_seq = nt_seq[start - 1:stop] 
        nt_seq = str(Seq(nt_seq).reverse_complement()) if (strand == -1) else nt_seq # If on the opposite strand, get the reverse complement. 

        if( len(nt_seq) % 3 == 0) and (error == 'raise'):
            raise Exception(f'GBFFFile.get_nt_seq: Expected the length of the nucleotide sequence to be divisible by three, but sequence is of length {len(nt_seq)}.')

        return nt_seq

    def get_stop_codon(self, start:int=None, stop:int=None, strand:int=None, contig_id:int=None, **kwargs) -> str:
        return self.get_nt_seq(start=start, stop=stop, strand=strand, contig_id=contig_id)[-3:]
    
    def get_start_codon(self, start:int=None, stop:int=None, strand:int=None, contig_id:int=None, **kwargs) -> str:
        return self.get_nt_seq(start=start, stop=stop, strand=strand, contig_id=contig_id)[:3]

    def add_start_stop_codons(self, df:pd.DataFrame) -> pd.DataFrame:
        '''Add start and stop codons to the input DataFrame. Assumes the DataFrame contains, at least, columns for the 
        nucleotide start and stop positions, the strand, and the contig ID.'''
        start_codons, stop_codons = ['ATG', 'GTG', 'TTG'], ['TAA', 'TAG', 'TGA']

        df['stop_codon'] = [self.get_stop_codon(**row) for row in df.to_dict(orient='records')]
        df['start_codon'] = [self.get_start_codon(**row) for row in df.to_dict(orient='records')]

        # df.stop_codon = df.stop_codon.apply(lambda c : 'none' if (c not in stop_codons) else c)
        # df.start_codon = df.start_codon.apply(lambda c : 'none' if (c not in start_codons) else c)

        return df

