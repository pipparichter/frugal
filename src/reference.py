from src.files import GBFFFile, FASTAFile
import pandas as pd 
import numpy as np
from tqdm import tqdm 
from src import get_genome_id, fillna
import warnings
import os 
from Bio.Align import PairwiseAligner
import re

# TODO: Take a closer look at this file, GCF_000009085.1_genomic.gbff, which seems to have a lot of weirdness. It seems as though 
#   pseudogenes are being entered as misc_features.  
# TODO: Check to make sure that the sequences I have predicted as conflicts are actually conflicts. I am worried that there are some cases where
#   there are overlapping genes, and the Prodigal-predicted sequence is in-frame with one of the genes, but because of mis-predicted gene boundaries,
#   it shares the most overlap with the wrong gene. Update: This does seem to happen in some cases, e.g. NP_214608.1

class Reference():

    query_fields = ['start', 'stop', 'partial', 'strand', 'seq', 'gc_content', 'rbs_motif', 'rbs_spacer', 'start_type']
    subject_fields = ['start', 'stop', 'partial', 'strand', 'seq']


    def __init__(self, path:str):

        self.genome_id = get_genome_id(path)
        df = GBFFFile(path).to_df()
        df['genome_id'] = self.genome_id
        self.df = df

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
        top_hits_df = self.df[self.df.contig_id == query.contig_id] # Get the contig corresponding of the query region. 
        # The and stop are both inclusive, so if starts or stops are equal, it counts as an overlap.  
        hits_df = top_hits_df[~(top_hits_df.start > query.stop) & ~(top_hits_df.stop < query.start)].copy() # Everything which passes this filter overlaps with the query region. 
        
        if len(hits_df) == 0:
            return None

        hits_info_df = list()

        for subject in hits_df.itertuples():
            hit = {'query_id':query.Index, 'subject_id':subject.Index}
            hit.update({f'query_{field}':getattr(query, field, None) for field in Reference.query_fields})
            hit.update({f'subject_{field}':getattr(subject, field) for field in Reference.subject_fields})  
            hit['subject_length'] = (subject.stop - subject.start) + 1
            hit['query_length'] = (query.stop - query.start) + 1
            hit['same_strand'] = (subject.strand == query.strand) 
            hit['overlap_start'] = max(subject.start, query.start)   
            hit['overlap_stop'] = min(subject.stop, query.stop)
            hit['overlap_length'] = (hit['overlap_stop'] - hit['overlap_start']) + 1 # Add a one to account for the fact that bounds are inclusive.
            hit['subject_overlap_fraction'] = hit['overlap_length'] / hit['subject_length'] # Add a one to account for the fact that bounds are inclusive.
            hit['query_overlap_fraction'] = hit['overlap_length'] / hit['query_length']# Add a one to account for the fact that bounds are inclusive.
            hit['exact_match'] = (query.start == subject.start) and (query.stop == subject.stop)
            hit.update(Reference.is_in_frame(query, subject))
            hits_info_df.append(hit)
        hits_info_df = pd.DataFrame(hits_info_df)
        
        hits_df.columns = ['subject_' + col for col in hits_df.columns]
        hits_df = hits_df.drop(columns=hits_info_df.columns, errors='ignore') # Drop shared columns to make sure they aren't duplicated by the merge. 
        hits_df = hits_df.merge(hits_info_df, left_index=True, right_on='subject_id', validate='one_to_one') # This will reset the index, which is no longer important. 
        return hits_df

    def compare(self, query_df:pd.DataFrame, verbose:bool=True):
        all_hits_df = list()
        for query in tqdm(list(query_df.itertuples()), desc='Reference.compare', disable=(not verbose)):
            hits_df = self._get_hits(query) # Get the hit with the biggest overlap, with a preference for "valid" hits.
            if hits_df is not None:
                all_hits_df.append(hits_df)

        all_hits_df = pd.concat(all_hits_df).reset_index(drop=True)
        top_hits_df = Reference._get_top_hits(query_df, all_hits_df) 
        return all_hits_df, top_hits_df

    @staticmethod
    def _get_top_hits(query_df:pd.DataFrame, all_hits_df:pd.DataFrame):
        top_hits_df = []
        for query_id, df in all_hits_df.groupby('query_id'):
            row = dict()
            row['query_id'] = query_id
            row['n_hits'] = len(df)
            row['n_hits_same_strand'] = df.same_strand.sum()
            row['n_hits_opposite_strand'] = len(df) - row['n_hits_same_strand']
            row['n_hits_in_frame'] = df.in_frame.sum()
            # Sort values on a boolean will put False (0) first, and True (1) last if ascending is True. 
            # Want to prioritize hits that are in-frame. 
            top_hit = df.sort_values(by=['in_frame', 'overlap_length'], ascending=False).iloc[0]
            top_hit = {field.replace('subject_', 'top_hit_'):value for field, value in top_hit.to_dict().items()}
            row.update(top_hit)
            top_hits_df.append(row)
        top_hits_df = pd.DataFrame(top_hits_df).set_index('query_id')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            # Add the query IDs which had no compare hits to the DataFrame. Make sure to include the sequences.  
            index = pd.Index(name='query_id', data=[id_ for id_ in query_df.index if (id_ not in top_hits_df.index)])
            no_hits_df = pd.DataFrame(index=index, columns=top_hits_df.columns)
            no_hits_df['query_seq'] = query_df['seq'].loc[no_hits_df.index].copy()
            top_hits_df = pd.concat([top_hits_df, no_hits_df])
            top_hits_df = top_hits_df.loc[query_df.index] # Make sure the DataFrames are in the same order for convenience.

        return fillna(top_hits_df, rules={bool:False, str:'none', int:0, float:0}, errors='raise')
    
    @staticmethod
    def load(path:str) -> pd.DataFrame:
        dtypes = {'top_hit_partial':str, 'query_partial':str, 'top_hit_translation_table':str, 'top_hit_codon_start':str}
        dtypes.update({'top_hit_pseudo':bool, 'in_frame':bool})
        df = pd.read_csv(path, dtype=dtypes, index_col=0) # Load in the reference output. 
        return df 



class ReferenceAnnotator():
    '''Assigns one of the following categories to a sequence based on the results of comparing its coordinates to the 
    reference genome annotation. 
    
    (1) match: Sequences where the boundaries exactly match the reference, or have 100 percent sequence identity with the reference. 
    (2) conflict: Sequences which exceed max_overlap with any 
    (3) intergenic
    (4) pseudogene: A separate category is needed for pseudogenes, because it is hard to determine if it is a match or spurious
        translation in these cases. This is the category for any predictions with same-strand overlap with a pseudogene.  
    '''

    categories = np.array(['match', 'intergenic', 'conflict', 'pseudogene'])

    # is_hypothetical = lambda df : df.top_hit_product == 'hypothetical protein'
    # is_ab_initio = lambda df : df.top_hit_evidence_type == 'ab initio prediction'
    # is_suspect = lambda df : ReferenceAnnotator.is_hypothetical(df) & ReferenceAnnotator.is_ab_initio(df) # This will be False for intergenic sequences. 

    def __init__(self, max_overlap:int=50, min_sequence_identity:float=1):

        self.max_overlap = max_overlap
        self.min_sequence_identity = min_sequence_identity

        self.is_pseudogene = lambda df : (df.same_strand & df.top_hit_pseudo)
        self.is_match = lambda df : ~self.is_pseudogene(df) & df.in_frame & (df.top_hit_feature == 'CDS')
        self.is_intergenic = lambda df :  ~self.is_match(df) & ~self.is_pseudogene(df) & (df.overlap_length < max_overlap) # Does not overlap with anything. 
        self.is_conflict = lambda df : ~self.is_intergenic(df) & ~self.is_match(df) & ~self.is_pseudogene(df)  

    @staticmethod
    def _get_sequence_identity(seq:str, top_hitseq:str) -> float: 
        aligner = PairwiseAligner(match_score=1, mismatch_score=0, gap_score=0)
        alignment = aligner.align(seq, top_hitseq)[0] # I think this will get the best alignment?
        score = alignment.score
        score = max(score / len(seq), score / len(top_hitseq)) # Normalize the score by sequence length. 
        return score
    
    def _check(self, top_hits_df:pd.DataFrame):

        top_hits_df['sequence_identity'] = np.where(top_hits_df.exact_match, 1, 0).astype(np.float64)
        
        mask, n_downgraded = ((top_hits_df.category == 'match') & ~top_hits_df.exact_match), 0
        for row in tqdm(top_hits_df[mask].itertuples(), total=mask.sum(), desc='ReferenceAnnotator._check'):
            sequence_identity = ReferenceAnnotator._get_sequence_identity(row.query_seq, row.top_hit_seq)
            top_hits_df.loc[row.Index, 'sequence_identity'] = sequence_identity
            if (sequence_identity < self.min_sequence_identity):
                top_hits_df.loc[row.Index, 'category'] = 'conflict' if (row.overlap_length >= self.max_overlap) else 'intergenic'
                n_downgraded += 1


        # I think there are some cases which are matches, but because one sequence is partial, they are not registering
        # as in-frame. These are being categorized as conflicts or intergenic depending on the overlap. 

        mask, n_upgraded = ((top_hits_df.category.isin(['intergenic', 'conflict'])) & top_hits_df.same_strand & (top_hits_df.top_hit_feature == 'CDS')), 0
        for row in tqdm(top_hits_df[mask].itertuples(), total=mask.sum(), desc='ReferenceAnnotator._check'):
            sequence_identity = ReferenceAnnotator._get_sequence_identity(row.query_seq, row.top_hit_seq)
            top_hits_df.loc[row.Index, 'sequence_identity'] = sequence_identity
            if (sequence_identity >= self.min_sequence_identity):
                top_hits_df.loc[row.Index, 'category'] = 'match'
                n_upgraded += 1
            
        print(f'ReferenceAnnotator._check: Downgraded {n_downgraded} "match" sequences to "intergenic" or "conflict".')
        print(f'ReferenceAnnotator._check: Upgraded {n_upgraded} "intergenic" or "conflict" sequences to "match.')

        return top_hits_df 

    def run(self, path:str):
        top_hits_df = Reference.load(path)
        conditions = [self.is_match(top_hits_df), self.is_intergenic(top_hits_df), self.is_conflict(top_hits_df), self.is_pseudogene(top_hits_df)]
        top_hits_df['category'] = np.select(conditions, ReferenceAnnotator.categories, default='none')
        assert (top_hits_df.category == 'none').sum() == 0, 'ReferenceAnnotator.run: Some sequences were not assigned annotations.'

        top_hits_df['sequence_identity'] = 0.0
        top_hits_df = self._check(top_hits_df)
        top_hits_df.to_csv(path) # Write the DataFrame back to the original path. 


def compare(query_path:str, reference_path:str, results_dir:str='../data/compare', annotate:bool=True, overwrite:bool=False, min_sequence_identity:float=0.9, max_overlap:int=50):

    genome_id = get_genome_id(query_path)

    all_hits_output_path = os.path.join(results_dir, f'{genome_id}_all_hits.csv')
    top_hits_output_path = os.path.join(results_dir, f'{genome_id}_top_hits.csv')

    if (not os.path.exists(top_hits_output_path)) or overwrite:
        reference = Reference(reference_path)
        query_df = FASTAFile(path=query_path).to_df(prodigal_output=True)
        all_hits_df, top_hits_df = reference.compare(query_df, verbose=False)
        all_hits_df.to_csv(all_hits_output_path)
        top_hits_df.to_csv(top_hits_output_path)
    if annotate:
        annotator = ReferenceAnnotator(max_overlap=max_overlap, min_sequence_identity=min_sequence_identity)
        annotator.run(top_hits_output_path)
    print(f'compare: Reference comparison complete. Results written to {results_dir}')



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

