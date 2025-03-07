from src.files import GBFFFile, FASTAFile
import pandas as pd 
import numpy as np
from tqdm import tqdm 
from src import get_genome_id, fillna
import warnings
import os 
import re

# TODO: Take a closer look at this file, GCF_000009085.1_genomic.gbff, which seems to have a lot of weirdness. It seems as though 
#   pseudogenes are being entered as misc_features.  


class ReferenceGenome():

    # feature_order = GBFFFile.coding_features + GBFFFile.noncoding_features 
    # feature_order += [feature for feature in GBFFFile.features if feature not in GBFFFile.coding_features + GBFFFile.noncoding_features]
    # feature_order = feature_order[::-1] # So that CDS is "biggest" in terms of importance. 

    def __init__(self, path:str, load_homologs:bool=True, homologs_dir:str='../data/proteins/homologs/'):

        self.genome_id = get_genome_id(path)
        df = GBFFFile(path).to_df()
        df['genome_id'] = self.genome_id
        self.df = df

        if load_homologs:
            self.load_homologs(dir_=homologs_dir)

    def __str__(self):
        return self.genome_id
    
    def __len__(self):
        return len(self.df)

    @staticmethod
    def is_in_frame(query, subject):
        info = {'in_frame':False, 'in_frame_c_terminus':False, 'in_frame_n_terminus':False}

        # This only makes sense to check if the subject sequence is coding.
        if subject.feature != 'CDS':
            return info
        if not (query.strand == subject.strand):
            return info 

        # The codon_start qualifier is relative to the translational start position, not the gene start position; adjusting based
        # on the specified offset therefore depends on the strand. 

        # codon_start can be either 1 (indicating no offset), 2, or 3. 

        # Must account for edge cases where the sequence is partial. Prodigal all edge sequences as partial, even if there is a valid start, 
        # so only use the subject sequence to check if partial. 
        if query.strand == 1:
            subject_start = subject.start + (int(subject.codon_start) - 1)
            query_start = query.start
            query_stop, subject_stop = query.stop, subject.stop
        elif query.strand == -1:
            subject_start = subject.stop - (int(subject.codon_start) - 1)
            query_start = query.stop
            query_stop, subject_stop = query.start, subject.start

        # There are cases where programmed frameshift means that the C-terminal part of the sequence is aligned, but the N-terminal
        # side is not. I still don't want to include these cases as spurious, as they are biologically-interesting. So I think
        # I should consider anything that is in-frame at either terminus as "in frame"
        info['in_frame_c_terminus'] = ((query_stop - subject_stop) % 3) == 0 
        info['in_frame_n_terminus'] = ((query_start - subject_start) % 3) == 0 
        info['in_frame'] = (info['in_frame_c_terminus'] or info['in_frame_n_terminus'])
        
        return info

    def _get_hits(self, query):

        ref_df = self.df[self.df.contig_id == query.contig_id] # Get the contig corresponding of the query region. 
        # The and stop are both inclusive, so if starts or stops are equal, it counts as an overlap.  
        hits_df = ref_df[~(ref_df.start > query.stop) & ~(ref_df.stop < query.start)].copy() # Everything which passes this filter overlaps with the query region. 
        
        if len(hits_df) == 0:
            return None

        hits_info_df = list()

        for subject in hits_df.itertuples():
            hit = {'query_id':query.Index, 'subject_id':subject.Index}
            hit.update({f'query_{field}':getattr(query, field) for field in ['start', 'stop', 'partial', 'strand', 'seq']})
            hit.update({f'subject_{field}':getattr(subject, field) for field in ['start', 'stop', 'partial', 'strand', 'seq']})  
            hit['subject_length'] = (subject.stop - subject.start) + 1
            hit['query_length'] = (query.stop - query.start) + 1
            hit['same_strand'] = (subject.strand == query.strand) 
            hit['overlap_start'] = max(subject.start, query.start)   
            hit['overlap_stop'] = min(subject.stop, query.stop)
            hit['overlap_length'] = (hit['overlap_stop'] - hit['overlap_start']) + 1 # Add a one to account for the fact that bounds are inclusive.
            hit['subject_overlap_fraction'] = hit['overlap_length'] / hit['subject_length'] # Add a one to account for the fact that bounds are inclusive.
            hit['query_overlap_fraction'] = hit['overlap_length'] / hit['query_length']# Add a one to account for the fact that bounds are inclusive.
            hit['exact_match'] = (query.start == subject.start) and (query.stop == subject.stop)
            hit.update(ReferenceGenome.is_in_frame(query, subject))
            hits_info_df.append(hit)
        hits_info_df = pd.DataFrame(hits_info_df)
        
        hits_df.columns = ['subject_' + col for col in hits_df.columns]
        hits_df = hits_df.drop(columns=hits_info_df.columns, errors='ignore') # Drop shared columns to make sure they aren't duplicated by the merge. 
        hits_df = hits_df.merge(hits_info_df, left_index=True, right_on='subject_id', validate='one_to_one') # This will reset the index, which is no longer important. 
        return hits_df

    def search(self, query_df:pd.DataFrame, verbose:bool=True, summarize:bool=True):
        results_df = list()
        for query in tqdm(list(query_df.itertuples()), desc='ReferenceGenome.search', disable=(not verbose)):
            hits_df = self._get_hits(query) # Get the hit with the biggest overlap, with a preference for "valid" hits.
            if hits_df is not None:
                results_df.append(hits_df)

        results_df = pd.concat(results_df).reset_index(drop=True)
        summary_df = ReferenceGenome.summarize(query_df, results_df) if summarize else None
        return results_df, summary_df

    @staticmethod
    def summarize(query_df:pd.DataFrame, results_df:pd.DataFrame):
        summary_df = []
        for query_id, df in results_df.groupby('query_id'):
            row = dict()
            row['query_id'] = query_id
            row['n_hits'] = len(df)
            row['n_hits_same_strand'] = df.same_strand.sum()
            row['n_hits_opposite_strand'] = len(df) - row['n_hits_same_strand']
            row['n_hits_in_frame'] = df.in_frame.sum()
            # Sort values on a boolean will put False (0) first, and True (1) last if ascending is True. 
            top_hit = df.sort_values(by=['overlap_length', 'in_frame'], ascending=False).iloc[0]
            top_hit = {field.replace('subject_', 'top_hit_'):value for field, value in top_hit.to_dict().items()}
            row.update(top_hit)
            summary_df.append(row)
        summary_df = pd.DataFrame(summary_df).set_index('query_id')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            # Add the query IDs which had no search results to the summary DataFrame. 
            index = pd.Index(name='query_id', data=[id_ for id_ in query_df.index if (id_ not in summary_df.index)])
            summary_df = pd.concat([summary_df, pd.DataFrame(index=index, columns=summary_df.columns)])
            summary_df = summary_df.loc[query_df.index] # Make sure the DataFrames are in the same order for convenience.

        return fillna(summary_df, rules={bool:False, str:'none', int:0, float:0}, errors='raise')

    def load_homologs(self, dir_:str='../data/proteins/homologs/'):

        protein_id_pattern = ['UniRef([0-9]{1,})_([0-9A-Za-z]+)', '[A-Z]P_([0-9]+).([0-9]+)']
        protein_id_pattern = re.compile(f"({'|'.join(protein_id_pattern)})")

        path = os.path.join(dir_, f'{self.genome_id}_protein.faa')
        if not os.path.exists(path):    
            print(f'\nReferenceGenome.load_homologs: No homolog file for genome {self.genome_id} was found in {dir_}')
            return
        
        homologs_df = FASTAFile(path=path).to_df(prodigal_output=False)
        homologs_df = homologs_df.groupby(homologs_df.index).first() # Drop any duplicates. 
        print(f'\nReferenceGenome.add_homologs: Loaded {len(homologs_df)} homologs; {self.df.pseudo.sum()} pseudogenes present in the genome.')
        homologs_df.index.name = 'evidence_details'
        homologs_df, _ = homologs_df.align(self.df.set_index('evidence_details'), axis=0, join='right', fill_value='none')
        assert len(homologs_df) == len(self.df), f'ReferenceGenome.load_homologs: Expected len(homologs_df) == len(self.df), but len(homologs_df) is {len(homologs_df)} and len(self.df) is {len(self.df)}.'
        self.df['homolog_seq'] = homologs_df.seq.values
        self.df['homolog_id'] = ['none' if (re.match(protein_id_pattern, id_) is None) else id_ for id_ in homologs_df.index]




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

