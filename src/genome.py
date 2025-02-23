from src.files.gbff import GBFFFile
import pandas as pd 
import numpy as np
from Bio.Seq import Seq
from tqdm import tqdm 


class ReferenceGenome():

    hit_info = {'locus_tag':None, 'feature':None, 'in_frame_hit':False, 'start_aligned_hit':False, 'stop_aligned_hit':False, 'hit_overlap':0}

    def __init__(self, genome_id:str, database_path:str='../data/ncbi_database_cds.csv'):

        self.genome_id = genome_id
        df = pd.read_csv(database_path, index_col=0, dtype={'partial':str}, low_memory=False)
        self.df = df[df.genome_id == genome_id]

    def __str__(self):
        return self.genome_id
    
    def __len__(self):
        return len(self.df)

    def _merge(self, df:pd.DataFrame, hits_df:pd.DataFrame):
        n = len(df)

        df_no_hits = df[hits_df.n_valid_hits == 0].copy()
        # print(f'ReferenceGenome._merge: {len(df_no_hits)} entries in the query DataFrame had no valid hits in the reference.')
        
        hits_df = hits_df[hits_df.n_valid_hits > 0].rename(columns={'locus_tag':'ref_locus_tag', 'feature':'ref_feature'})
        ref_df = self.df.copy().rename(columns={col:'ref_' + col for col in self.df.columns})
        # The locus tags are not unique, so we can't use it to merge, also need to use the features... 
        hits_df = hits_df.merge(ref_df, on=['ref_locus_tag', 'ref_feature'], how='left').set_index(hits_df.index)

        df = hits_df.merge(df, left_index=True, right_index=True, how='left')        
        df.index.name = 'id' # Restore the index name, which is lost in the merging. 
        df = pd.concat([df, df_no_hits], ignore_index=False)

        # Not sure why, but these fill as NaNs after concatenating.
        df['n_hits'] = df.n_hits.fillna(0).astype(int)
        df['n_valid_hits'] = df.n_hits.fillna(0).astype(int)
        df['n_same_strand_hits'] = df.n_hits.fillna(0).astype(int)
        df['hit_overlap'] = df.n_hits.fillna(0).astype(int)

        assert (len(df) == n), f'ReferenceGenome._merge: There was a problem merging the query DataFrame with the search results.'
        return df

    @staticmethod
    def _add_hit_info(ref_df:pd.DataFrame, query):
        ref_df['same_strand_hit'] = (ref_df.strand == query.strand)
        ref_df['stop_aligned_hit'] = (ref_df.stop == query.stop)
        ref_df['start_aligned_hit'] = (ref_df.start == query.start)
        ref_df['in_frame_hit'] = ((ref_df.stop - query.stop) & 3 == 0) | ((ref_df.start - query.start) & 3 == 0) 
        ref_df['valid_hit'] = ref_df.same_strand_hit & (ref_df.start_aligned_hit | ref_df.stop_aligned_hit | ref_df.in_frame_hit)

        # Add a check to make sure the logic makes sense... 
        assert (ref_df.start_aligned_hit | ref_df.stop_aligned_hit).sum() <= ref_df.in_frame_hit.sum(), 'ReferenceGenome._get_hit_info: The number of in-frame hits should be at least the number of aligned hits.'
        # assert ref_df.start_aligned_hit.sum() <= ref.df_in_frame_hit.sum(), 'ReferenceGenome._get_hit_info: The number of in-frame hits should be at least the number of aligned hits.'
        # assert ref_df.stop_aligned_hit.sum() <= ref.df_in_frame_hit.sum(), 'ReferenceGenome._get_hit_info: The number of in-frame hits should be at least the number of aligned hits.'
        return ref_df

    def _get_hit(self, query):
        
        hit = {'n_hits':0, 'n_valid_hits':0, 'n_same_strand_hits':0} # Instantiate a hit. 

        ref_df = self.df.copy() 
        ref_df = ref_df[ref_df.contig_id == query.contig_id]
        ref_df = ref_df[~(ref_df.start > query.stop) & ~(ref_df.stop < query.start)] # Filter out everything which definitely has no overlap.

        # Case where there are no detected genes on a contig in the GBFF file, but Prodigal found one.
        if len(ref_df) == 0: 
            hit.update(ReferenceGenome.hit_info) # Add the default hit info to make datatypes consistent. 
            return hit

        ref_df = ReferenceGenome._add_hit_info(ref_df.copy(), query)

        hit['n_hits'] = len(ref_df)
        hit['n_valid_hits'] = ref_df.valid_hit.sum().item()
        hit['n_same_strand_hits'] = ref_df.same_strand_hit.sum().item()
        
        ref_df = ref_df[ref_df.valid_hit] # Filter by the valid hits. 
        
        if len(ref_df) > 0:
            # There are some cases with more than one valid hit, in which case we want to grab the hit with the biggest overlap. 
            # It might be interesting to look into these cases a bit more, but they are very rare. 
            ref_df['hit_overlap'] = ref_df.apply(lambda row : len(np.intersect1d(np.arange(row.start, row.stop), np.arange(query.start, query.stop))), axis=1)
            ref_df = ref_df.sort_values(by='hit_overlap', ascending=False) # Sort so that the best hit is at the top. 
            
            for field in ReferenceGenome.hit_info.keys(): # Get all relevant info for the top hit. 
                hit[field] = ref_df[field].iloc[0]
            return hit

        elif hit['n_valid_hits'] == 0:
            hit.update(ReferenceGenome.hit_info) # Add the default hit info. 
            return hit

    def search(self, df:pd.DataFrame, add_start_stop_codons:bool=True, verbose:bool=True):
        hits_df = list()
        for query in tqdm(list(df.itertuples()), desc='ReferenceGenome.search', disable=(not verbose)):
            hit = self._get_hit(query) # Get the hit with the biggest overlap, with a preference for "valid" hits. 
            hit['id'] = query.Index
            hits_df.append(hit)
        hits_df = pd.DataFrame(hits_df).set_index('id')
        df = self._merge(df, hits_df)
        return df


    # def _add_start_stop_codons(self):
    #     self.df = self.add_start_stop_codons(self.df)

    # def _add_lengths(self):
    #     # Can't just use seq.apply(len) because forcing the sequences to be strings causes null sequences (e.g., in the case of non-CDS features) to be 'nan'.'''
    #     # This also gets lengths for pseudogenes. 
    #     lengths = list()
    #     for row in self.df.itertuples():
    #         if (row.feature == 'CDS'):
    #             lengths.append((row.stop - row.start) // 3) # The start and stop indices are in terms of nucleotides. 
    #         else:
    #             lengths.append(None)
    #     self.df['length'] = lengths 


    # def get_nt_seq(self, start:int=None, stop:int=None, strand:int=None, contig_id:int=None, error:str='ignore'):
    #     nt_seq = self.contigs[contig_id] 
    #     # Pretty sure the stop position is non-inclusive, so need to shift it over.
    #     nt_seq = nt_seq[start - 1:stop] 
    #     nt_seq = str(Seq(nt_seq).reverse_complement()) if (strand == -1) else nt_seq # If on the opposite strand, get the reverse complement. 

    #     if( len(nt_seq) % 3 == 0) and (error == 'raise'):
    #         raise Exception(f'GBFFFile.get_nt_seq: Expected the length of the nucleotide sequence to be divisible by three, but sequence is of length {len(nt_seq)}.')

    #     return nt_seq

    # def get_stop_codon(self, start:int=None, stop:int=None, strand:int=None, contig_id:int=None, **kwargs) -> str:
    #     return self.get_nt_seq(start=start, stop=stop, strand=strand, contig_id=contig_id)[-3:]
    
    # def get_start_codon(self, start:int=None, stop:int=None, strand:int=None, contig_id:int=None, **kwargs) -> str:
    #     return self.get_nt_seq(start=start, stop=stop, strand=strand, contig_id=contig_id)[:3]

    # def add_start_stop_codons(self, df:pd.DataFrame) -> pd.DataFrame:
    #     '''Add start and stop codons to the input DataFrame. Assumes the DataFrame contains, at least, columns for the 
    #     nucleotide start and stop positions, the strand, and the contig ID.'''
    #     start_codons, stop_codons = ['ATG', 'GTG', 'TTG'], ['TAA', 'TAG', 'TGA']

    #     df['stop_codon'] = [self.get_stop_codon(**row) for row in df.to_dict(orient='records')]
    #     df['start_codon'] = [self.get_start_codon(**row) for row in df.to_dict(orient='records')]

    #     # df.stop_codon = df.stop_codon.apply(lambda c : 'none' if (c not in stop_codons) else c)
    #     # df.start_codon = df.start_codon.apply(lambda c : 'none' if (c not in start_codons) else c)

    #     return df

