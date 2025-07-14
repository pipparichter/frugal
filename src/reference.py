from src.files import GBFFFile, FASTAFile
import pandas as pd 
import numpy as np
from tqdm import tqdm 
from src import get_genome_id, fillna
import warnings
import os 
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq
from Bio.Data.CodonTable import TranslationError
import re
from collections import namedtuple
import itertools

reverse_complement = lambda seq : str(Seq(seq).reverse_complement()).replace('T', 'U')

OpenReadingFrame = namedtuple('OpenReadingFrame', ['start', 'stop', 'seq'])
Query = namedtuple('Query', ['Index', 'contig_id', 'start', 'stop', 'strand'])

start_codons = {11:['AUG', 'GUG', 'UUG'], 14:['AUG', 'GUG', 'UUG']}
stop_codons = {11:['UAA', 'UAG', 'UGA'], 14:['UAG', 'UGA']} # In code 14, UAA codes for tyrosine. 

# This function does not filter out potential ORFs with in-frame stops. 
is_valid_orf = lambda start, stop : (start < stop) and (((stop - start) % 3) == 0)
is_valid_cds = lambda nt_seq : (len(nt_seq) % 3 == 0) and (nt_seq[:3] in start_codons[11]) and (nt_seq[-3:] in stop_codons[11])

get_contig_id = lambda id_ : id_.split('.')[0]

has_overlap = lambda query_start, query_stop, subject_start, subject_stop : not ((query_stop < subject_start) or (query_start > subject_stop))

def get_overlap_type(query_start:int=None, query_stop:int=None, subject_start:int=None, subject_stop:int=None, query_strand:int=None, subject_strand:int=None, **kwargs):
    if not has_overlap(query_start, query_stop, subject_start, subject_stop):
        overlap_type = 'none'
        
    elif (query_start >= subject_start) and (query_stop <= subject_stop):
        overlap_type = 'nested'
    elif (query_start <= subject_start) and (query_stop >= subject_stop):
        overlap_type = 'nested'
    elif (subject_strand == query_strand):
        overlap_type = 'tandem'

    elif (query_strand == 1) and ((query_stop >= subject_start) & (query_stop <= subject_stop)):
        overlap_type = 'convergent' 
    elif (query_strand == -1) and ((query_start >= subject_start) & (query_start <= subject_stop)):
        overlap_type = 'convergent' 
    elif (query_strand == 1) and ((query_start >= subject_start) & (query_start <= subject_stop)):
        overlap_type = 'divergent' 
    elif (query_strand == -1) and ((query_stop >= subject_start) & (query_stop <= subject_stop)):
        overlap_type = 'divergent' 

    return overlap_type


class Reference():

    query_fields = ['start', 'stop', 'partial', 'strand', 'seq', 'gc_content', 'rbs_motif', 'rbs_spacer', 'start_type']
    subject_fields = ['start', 'stop', 'partial', 'strand', 'seq']



    def __init__(self, path:str, load_contigs:bool=False, translation_table:int=11):

        self.genome_id = get_genome_id(path, errors='ignore', default=os.path.basename(path).replace('_genomic.gbff', ''))
        file = GBFFFile(path)

        self.df = file.to_df()
        self.df['genome_id'] = self.genome_id

        self.contigs = file.contigs if load_contigs else None # Maps contig IDs to nucleotides. 
        self.translation_table = translation_table
        self.start_codons = start_codons[translation_table]
        self.stop_codons = stop_codons[translation_table]

    def __str__(self):
        return self.genome_id
    
    def __len__(self):
        return len(self.df)
    
    def get_nt_seq(self, contig_id:str, start:int, stop:int):
        contig = self.contigs[contig_id]
        return contig[start:stop]
    
    def _get_codon_idxs(self, seq:str, codons:list=None):
        pattern = f"(?=({'|'.join(codons)}))" # Use lookahead assertion to allow for overlapping matches. 
        # Use finditer instead of findall to return the start positions of each match. 
        return [match.start(0) for match in re.finditer(pattern, seq)]
    
    def _get_potential_orfs(self, seq:str):

        orfs = list()

        idxs = list()
        idxs.append(self._get_codon_idxs(seq, self.start_codons))
        idxs.append(self._get_codon_idxs(seq, self.stop_codons))

        total = len(idxs[0]) * len(idxs[1]) # Total number of potential ORFs to scan.
        # Do an exhaustive search of every start-stop codon pair. 
        for start, stop in tqdm(itertools.product(*idxs), desc='Reference._get_potential_orfs', total=total):
            if is_valid_orf(start, stop):
                orfs.append(OpenReadingFrame(start, stop + 3, seq[start:stop + 3]))

        print(f'Reference._get_potential_orfs: Found {len(orfs)} potential ORFs out of {total} start-stop codon pairs.')
        return orfs 
        
    def _get_orf_translation(self, orf:OpenReadingFrame):
        # With cds=True, will raise an exception if there is an in-frame stop, or the sequence is not a multiple of three. 
        seq = str(Seq(orf.seq).translate(table=self.translation_table, stop_symbol='', cds=True))
        return seq 
    
    @staticmethod
    def _get_orf_coordinate(orf:OpenReadingFrame, seq_start:int=None, seq_stop:int=None, strand:int=None):
        '''Convert the start and stop indices for an ORF, which are defined relative to the sub-sequence which was searched, back to absolute
        coordinates (relative to the start and stop of the entire contig). Also shift the start position right by one, as the convention 
        for gene coordinates is inclusive, one-indexed boundaries.'''
        # If I remember correctly, both bounds are inclusive, and the start position is one-indexed. 
        if strand == 1:
            start = seq_start + orf.start # Both of these should be zero-indexed. 
            stop = start + (orf.stop - orf.start)
        elif strand == -1:
            stop = seq_stop - orf.start
            start = stop - (orf.stop - orf.start)
        return (start + 1, stop) 
    
    def get_contig_ids(self):
        return list(self.contigs.keys())
    
    def get_orfs(self, contig_id:str, start:int=None, stop:int=None, min_length:int=50):
        
        seq = self.contigs[contig_id][start:stop]
        seq = seq.upper().replace('T', 'U')

        orfs = dict()
        orfs[1] = self._get_potential_orfs(seq)
        orfs[-1] = self._get_potential_orfs(reverse_complement(seq))

        orf_df = list()
        for strand, orfs_ in orfs.items():
            for orf in orfs_:
                try: # This should throw an error if the ORF could not be correctly-translated as a CDS. 
                    row = dict()
                    row['seq'] = self._get_orf_translation(orf)
                    row['start'], row['stop'] = self._get_orf_coordinate(orf, seq_start=start, seq_stop=stop, strand=strand)
                    row['start_codon'] = orf.seq[:3]
                    row['stop_codon'] = orf.seq[-3:]
                    row['strand'] = strand
                    row['length'] = len(row['seq'])
                    orf_df.append(row)
                except TranslationError as err:
                    pass
                    # print(f'Reference.get_orfs: Translation error, "{err}"')
        orf_df = pd.DataFrame(orf_df, index=pd.Index([f'{contig_id}_ORF_{i}' for i in range(len(orf_df))], name='id'))
        if min_length is not None:
            orf_df = orf_df[orf_df.length >= min_length].copy()
        return orf_df
    
    @staticmethod
    def _get_translation_start_stop(start:int=None, stop:int=None, strand:int=None, codon_start:int=1, adjust_start:bool=True, check:bool=False, **kwargs):
        # The codon_start qualifier is relative to the translational start position, not the gene start position; adjusting based
        # on the specified offset therefore depends on the strand. codon_start can be either 1 (indicating no offset), 2, or 3.

        # When determinining phase difference, both stop and stop coordinates must be inclusive (adjust_start=False). 
        # Otherwise, in the case of antisense overlaps, would be comparing a non-inclusive boundary to an inclusive boundary. 
        start = start - 1 if adjust_start else start 

        if strand == -1:
            start, stop = stop - (int(codon_start) - 1), start
        elif strand == 1:
            start, stop = start + (int(codon_start) - 1), stop
        
        if check: # This fails if a top hit sequence is partial at the C-terminus, or in the case of a programmed frameshift.. 
            length = abs(stop - start) if adjust_start else abs(stop - start) + 1
            assert length % 3 == 0, f'Reference._get_translational_start_stop: Sequence length should be divisible by three, but got {length}.'

        return start, stop 

    @staticmethod
    def _get_frame_info(query, subject):
        info = {'in_frame':False}
        
        is_in = lambda coordinate, start, stop : (coordinate >= start) and (coordinate <= stop)
        
        subject_overlap, query_overlap = ['0', '0'], ['0', '0']
        subject_overlap[0] = str(int(is_in(subject.start, query.start, query.stop)))
        subject_overlap[1] = str(int(is_in(subject.stop, query.start, query.stop)))
        query_overlap[0] = str(int(is_in(query.start, subject.start, subject.stop)))
        query_overlap[1] = str(int(is_in(query.stop, subject.start, subject.stop)))
        info['query_overlap'] = ''.join(query_overlap)
        info['subject_overlap'] = ''.join(subject_overlap)

        # This only makes sense to check if the subject sequence is coding.
        if (subject.feature != 'CDS') or (subject.pseudo):
            return info

        query_start, _ = Reference._get_translation_start_stop(adjust_start=False, check=True, **query._asdict())
        subject_start, _ = Reference._get_translation_start_stop(adjust_start=False, check=False, **subject._asdict())
        # subject_start, _ = Reference._get_translation_start_stop(adjust_start=False, check=(subject.partial == '00'), **subject._asdict())

        # Prodigal start and stop indices always correspond to the translational bounds, so always correspond to a nucleotide sequence with a length divisible by three. 
        # The same is not true for the top hit sequences, but these will always specify a codon_start, which is the offset to the start of translation. 
        # Therefore, the safe thing to do is to just compare translational starts (accounting for codon_start). This supports match cases where a programmed frameshift
        # causes the C terminus to be out-of-phase.
        info['phase'] = abs(query_start - subject_start) % 3


        # There are cases where programmed frameshift means that the C-terminal part of the sequence is aligned, but the N-terminal
        # info['in_frame'] = ((info['phase_start'] == 0) or (info['phase_stop'] == 0)) and (query.strand == subject.strand)
        info['in_frame'] = (info['phase'] == 0) and (query.strand == subject.strand)
        return info
    
    def get_hits(self, contig_id:str, start:int=None, stop:int=None, strand:int=None):
        query = Query(Index='none', start=start, contig_id=contig_id, stop=stop, strand=strand)
        return self._get_hits(query)
    
    def _get_nt_seqs(self, top_hits_df:pd.DataFrame, prefix:str='query'):
        assert self.contigs is not None, 'Reference._get_nt_seqs: Contigs have not been loaded into the Reference object.'

        nt_seqs = list()
        for row in top_hits_df.itertuples():
            
            contig_id = get_contig_id(row.Index) # Prodigal gene IDs contain the contig ID, e.g. NZ_CP130454.1_3215. 

            feature = 'CDS' if (prefix == 'query') else getattr(row, 'top_hit_feature')
            if feature != 'CDS':
                nt_seqs.append('none')
                continue 

            start, stop, strand, partial = getattr(row, f'{prefix}_start'), getattr(row, f'{prefix}_stop'), getattr(row, f'{prefix}_strand'), getattr(row, f'{prefix}_partial')

            nt_seq = self.get_nt_seq(contig_id, start - 1, stop).replace('T', 'U')
            nt_seq = reverse_complement(nt_seq) if (strand == -1) else nt_seq

            # if partial == '00': # Only check if the CDS is not partial.
            #     assert is_valid_cds(nt_seq), f'Reference._get_nt_seqs: {row} {strand} {prefix} {nt_seq} is not valid.'
            nt_seqs.append(nt_seq)

        top_hits_df[f'{prefix}_nt_seq'] = nt_seqs
        return top_hits_df            

    
    def _get_nt_seqs_upstream(self, top_hits_df:pd.DataFrame, prefix:str='query', n_upstream_bases:int=27):
        '''Default length of upstream region is based on the fact that the largest RBS spacer is 15 bp. We also want to include the start codon and the 5-bp SD sequence, 
        as well as a few base pairs upstream of SD. Also, the ribosome footprint is about 30 bp total (https://pubs.acs.org/doi/pdf/10.1021/acssynbio.2c00139?ref=article_openPDF#mk%3Aref14)'''

        assert self.contigs is not None, 'Reference._get_nt_seqs: Contigs have not been loaded into the Reference object.'

        is_partial_at_n_terminus = lambda row : ((getattr(row, f'{prefix}_partial') == '10') and (getattr(row, f'{prefix}_strand') == 1)) or ((getattr(row, f'{prefix}_partial') == '01') and (getattr(row, f'{prefix}_strand') == -1))
        # check_nt_seq_upstream = lambda nt_seq : (nt_seq[-3:] in start_codons[11] + start_codons[14]) and (len(nt_seq) == (n_upstream_bases + 3))

        nt_seqs = list()
        for row in top_hits_df.itertuples():
            
            contig_id = get_contig_id(row.Index) # Prodigal gene IDs contain the contig ID, e.g. NZ_CP130454.1_3215. 
            feature = 'CDS' if (prefix == 'query') else getattr(row, 'top_hit_feature')

            if feature != 'CDS':
                nt_seqs.append('none')
                continue 
            if is_partial_at_n_terminus(row):
                nt_seqs.append('partial')
                continue

            start, stop, strand = getattr(row, f'{prefix}_start'), getattr(row, f'{prefix}_stop'), getattr(row, f'{prefix}_strand')
            if strand == 1:
                start, stop = (start - 1) - (n_upstream_bases), (start - 1) + 3
            elif strand == -1:
                start, stop = stop - 3, stop  + n_upstream_bases 

            nt_seq = self.get_nt_seq(contig_id, start, stop).replace('T', 'U')
            nt_seq = reverse_complement(nt_seq) if (strand == -1) else nt_seq
            # assert check_nt_seq_upstream(nt_seq), f'Reference._get_upstream_nt_seqs: Something went wrong while grabbing the upstream region, {nt_seq}.'
            nt_seqs.append(nt_seq)

        top_hits_df[f'{prefix}_nt_seq_upstream'] = nt_seqs
        return top_hits_df      

    def _get_hits(self, query):
        hits_df = self.df[self.df.contig_id == query.contig_id] # Get the contig corresponding of the query region. 
        # The and stop are both inclusive, so if starts or stops are equal, it counts as an overlap.  
        hits_df = hits_df[~(hits_df.start > query.stop) & ~(hits_df.stop < query.start)].copy() # Everything which passes this filter overlaps with the query region. 
        
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
            hit['overlap_type'] = get_overlap_type(**hit)
            hit['overlap_length'] = (hit['overlap_stop'] - hit['overlap_start']) + 1 # Add a one to account for the fact that bounds are inclusive.
            hit['subject_overlap_fraction'] = hit['overlap_length'] / hit['subject_length'] # Add a one to account for the fact that bounds are inclusive.
            hit['query_overlap_fraction'] = hit['overlap_length'] / hit['query_length']# Add a one to account for the fact that bounds are inclusive.
            hit['exact_match'] = (query.start == subject.start) and (query.stop == subject.stop)
            hit.update(Reference._get_frame_info(query, subject))
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

        if self.contigs is not None:
            top_hits_df = self._get_nt_seqs(top_hits_df, prefix='top_hit')
            top_hits_df = self._get_nt_seqs(top_hits_df, prefix='query')
            top_hits_df = self._get_nt_seqs_upstream(top_hits_df, prefix='top_hit')
            top_hits_df = self._get_nt_seqs_upstream(top_hits_df, prefix='query')

        return all_hits_df, top_hits_df
    
    @staticmethod
    def _select_top_hit(df:pd.DataFrame):
        # Want to prioritize hits that are (1) in-frame and (2) are supported (i.e. not ab initio predictions). 
        df = df.sort_values(['in_frame', 'subject_supported', 'overlap_length'], ascending=False)
        top_hit = df.iloc[0]
        top_hit = {field.replace('subject_', 'top_hit_'):value for field, value in top_hit.to_dict().items()}
        return top_hit

    @staticmethod
    def _get_top_hits(query_df:pd.DataFrame, all_hits_df:pd.DataFrame):
        top_hits_df = []
        for query_id, df in all_hits_df.groupby('query_id'):
            df['subject_unsupported'] = (df.subject_product == 'hypothetical protein') & (df.subject_evidence_type == 'ab initio prediction')
            df['subject_supported'] = ~df.subject_unsupported 

            row = dict()
            row['query_id'] = query_id
            row['n_hits'] = len(df)
            row['n_hits_supported'] = df.subject_supported.sum()
            row['n_hits_same_strand'] = df.same_strand.sum()
            row['n_hits_opposite_strand'] = len(df) - row['n_hits_same_strand']
            row['n_hits_in_frame'] = df.in_frame.sum()
            top_hit = Reference._select_top_hit(df)
            row.update(top_hit)
            top_hits_df.append(row)
        top_hits_df = pd.DataFrame(top_hits_df).set_index('query_id')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            # Add the query IDs which had no compare hits to the DataFrame. Make sure to include the sequences.  
            no_hits_df = pd.DataFrame(index=pd.Index(name='query_id', data=[id_ for id_ in query_df.index if (id_ not in top_hits_df.index)]), columns=top_hits_df.columns)
            for field in Reference.query_fields: # Make sure all the query data is included. 
                no_hits_df[f'query_{field}'] = query_df.loc[no_hits_df.index][field]

            top_hits_df = pd.concat([top_hits_df, no_hits_df])
            top_hits_df = top_hits_df.loc[query_df.index] # Make sure the DataFrames are in the same order for convenience.

        return fillna(top_hits_df, rules={bool:False, str:'none', int:0, float:0}, errors='raise')
    
    @staticmethod
    def load(path:str) -> pd.DataFrame:
        dtypes = {'top_hit_partial':str, 'query_partial':str, 'top_hit_translation_table':str, 'top_hit_codon_start':str}
        dtypes.update({'top_hit_pseudo':bool, 'in_frame':bool})
        df = pd.read_csv(path, dtype=dtypes, index_col=0) # Load in the reference output. 
        return df 


class Aligner():

    def __init__(self, extend_gap_score:int=-1, open_gap_score:int=-11):

        self.extend_gap_score = extend_gap_score
        self.open_gap_score = open_gap_score
        self.aligner = PairwiseAligner(match_score=1, extend_gap_score=extend_gap_score, open_gap_score=open_gap_score)
        self.aligner.mode = 'local'

    # if permissive:
    #     matrix = substitution_matrices.load("PAM250")
    #     alphabet = set(matrix.alphabet)
    #     aligner.substitution_matrix = substitution_matrices.load("PAM250")
    #     query_seq = ''.join([aa if (aa in alphabet) else 'X' for aa in query_seq])
    #     subject_seq = ''.join([aa if (aa in alphabet) else 'X' for aa in subject_seq])

    @staticmethod
    def _get_alignment_strings(alignment, query_seq:str, subject_seq:str):
        '''Get string representations of the alignments from the returned alignment object.'''
        query_idxs, subject_idxs = alignment.aligned
        query_alignment, subject_alignment = '', ''

        # alignment.aligned returns a list of the form [[[idx, idx], [idx, idx], ...], [[idx, idx], [idx, idx], ...]]
        query_idx, subject_idx = 0, 0
        for (query_start, query_end), (subject_start, subject_end) in zip(query_idxs, subject_idxs):
            assert (query_end - query_start) == (subject_end - subject_start)
            if (query_idx > 0) and (subject_idx > 0): # Add gaps if it's not the start of the alignment.
                subject_gap_length = subject_start - subject_idx
                query_gap_length = query_start - query_idx
                max_gap_length = max(query_gap_length, subject_gap_length)

                query_alignment += '-' * (query_gap_length + (max_gap_length - query_gap_length))
                subject_alignment += '-' * (subject_gap_length + (max_gap_length - subject_gap_length))

            query_alignment += query_seq[query_start:query_end]
            subject_alignment += subject_seq[subject_start:subject_end]

            query_idx, subject_idx = query_end, subject_end

        return query_alignment, subject_alignment

    def _get_alignment_info(self, query_seq:str, subject_seq:str):

        info = {'n_matches':0, 'n_gap_opens':np.nan, 'raw_score':0, 'alignment_length':0, 'query_length':0, 'subject_length':0, 'query_alignment':'none', 'subject_alignment':'none',}
        
        if (subject_seq == 'none'):
            return info 

        alignment = self.aligner.align(query_seq, subject_seq)[0] # This is the best alignment. 
        
        info['alignment_length'] = alignment.length
        info['n_gap_opens'] = len(alignment.aligned[0])
        info['query_length'] = len(query_seq)
        info['subject_length'] = len(subject_seq)
        info['raw_score'] = alignment.score

        query_alignment, subject_alignment = Aligner._get_alignment_strings(alignment, query_seq, subject_seq)
        info['subject_alignment'] = subject_alignment
        info['query_alignment'] = query_alignment
        info['n_matches'] = sum(s == q for q, s in zip(query_alignment, subject_alignment) if (s != '-'))

        return info

    def align(self, df:pd.DataFrame, prefix:str='top_hit'):

        # pbar = tqdm(df.itertuples(), total=len(df), desc='Aligner.align')
        align_df = pd.DataFrame([self._get_alignment_info(row.query_seq, getattr(row, f'{prefix}_seq')) for row in df.itertuples()], index=df.index)
        align_df = align_df.rename(columns={col:col.replace('query_', '').replace('subject_', f'{prefix}_') for col in align_df.columns})
        align_df = align_df[[col for col in align_df.columns if (col not in df.columns)]] # Otherwise end up with two top_hit_length columns. 
        df = pd.concat([df, align_df], axis=1)
        df['sequence_identity'] = df.n_matches / df.alignment_length
        return df


def compare(query_path:str, reference_path:str):

    reference = Reference(reference_path, load_contigs=True)
    query_df = FASTAFile(path=query_path).to_df(prodigal_output=True)
    all_hits_df, top_hits_df = reference.compare(query_df, verbose=False)

    aligner = Aligner()
    all_hits_df = aligner.align(all_hits_df, prefix='subject')
    top_hits_df = aligner.align(top_hits_df)

    top_hits_df['genome_id'] = get_genome_id(query_path) 
    all_hits_df['genome_id'] = get_genome_id(query_path)

    assert len(top_hits_df) == len(query_df), 'compare: Length mismatch between the query DataFrame and output DataFrame.'

    return top_hits_df, all_hits_df


def is_pseudogene(df:pd.DataFrame, max_overlap_length:int=0, prefix:str='top_hit'):
    mask = df[f'{prefix}_pseudo'] & df.same_strand
    mask = mask & (df.overlap_length >= max_overlap_length)
    return mask 

# Based on inspection of the top hit reference sequences, some are clearly the same sequence but have less than 100% sequence identity. There seems to be
# a few different reasons for this:
# (1) The reference is recognized as partial, so the start codon is not M. 
# (2) The Prodigal sequence has more X's than the reference, perhaps because homology is used to compensate for an unknown nucleotide. 
# (3) There is a programmed frameshift that is missed by Prodigal. 

# There are two cases where the GeneMark prediction is apparently out-of-frame with the Prodigal predction, but sequence identity is still high
# (>80 %): NZ_JAALLS010000037.1_4 and NZ_AP035449.1_1439. In the case of NZ_JAALLS010000037.1_4, the alignment length is much longer than the overlap, suggesting possible translation
# of a repeat sequence (?). The other one is a mystery. 

def is_match(df:pd.DataFrame, min_sequence_identity:float=0.8, max_overlap_length:int=0, prefix:str='top_hit'):
    mask = (df[f'{prefix}_feature'] == 'CDS') & (~df[f'{prefix}_pseudo']) & df.in_frame
    mask = mask & (df.sequence_identity > min_sequence_identity)
    mask = mask & (df.overlap_length >= max_overlap_length)
    return mask
    
def is_intergenic(df:pd.DataFrame, max_overlap_length:int=0, prefix:str='top_hit'):
    mask = (df[f'{prefix}_feature'] == 'none') | ((df[f'{prefix}_feature'] == 'CDS') & (df.overlap_length < max_overlap_length))
    mask = mask & ~is_match(df, prefix=prefix) & ~is_pseudogene(df, prefix=prefix)
    return mask 
    
def is_conflict(df:pd.DataFrame, max_overlap_length:int=0, prefix:str='top_hit'):
    mask = ~is_match(df, max_overlap_length=max_overlap_length, prefix=prefix)
    mask = mask & ~is_pseudogene(df, max_overlap_length=max_overlap_length, prefix=prefix)
    mask = mask & ~is_intergenic(df, max_overlap_length=max_overlap_length, prefix=prefix)
    return mask 

def is_unsupported(df:pd.DataFrame, prefix:str='top_hit'):
    mask = (df[f'{prefix}_product'] == 'hypothetical protein')
    mask = mask & (df[f'{prefix}_evidence_type'] == 'ab initio prediction')
    return mask

def annotate(df:pd.DataFrame, prefix:str='top_hit'):
    df['pseudogene'] = is_pseudogene(df, prefix=prefix) 
    df['match'] = is_match(df, prefix=prefix)
    df['conflict'] = is_conflict(df, prefix=prefix)
    df['intergenic'] = is_intergenic(df, prefix=prefix)
    df['category'] = np.select([df.match, df.pseudogene, df.intergenic, df.conflict], ['match', 'pseudogene', 'intergenic', 'conflict'], default='none')
    assert (df.category == 'none').sum() == 0, 'Some of the sequences were not assigned a category.'
    
    if prefix == 'subject':
        df['length'] = df.query_seq.apply(len) # Make sure the lengths are in amino acids and not base pairs.
    else:
        df['length'] = df.seq.apply(len) # Make sure the lengths are in amino acids and not base pairs.

    df[f'{prefix}_length'] = np.where(df[f'{prefix}_seq'] == 'none', 0, df[f'{prefix}_seq'].apply(len)) # Make sure the lengths are in amino acids and not base pairs.
    df[f'{prefix}_unsupported'] = is_unsupported(df, prefix=prefix) 
    df['exact_match'] = df.match & (df[f'{prefix}_length'] == df.length) & (~df[f'{prefix}_ribosomal_slippage'])
    df['truncated'] = df.match & (df[f'{prefix}_length'] > df.length) & (~df[f'{prefix}_ribosomal_slippage'])
    df['extended'] = df.match & (df[f'{prefix}_length'] < df.length) & (~df[f'{prefix}_ribosomal_slippage'])

    return df.copy()


# def get_overlap_type(row):
#     kwargs = row.to_dict()
#     kwargs['subject_strand'] = row.top_hit_strand 
#     kwargs['subject_start'] = row.top_hit_start
#     kwargs['subject_stop'] = row.top_hit_stop
#     return Reference.get_overlap_type(**kwargs)


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

# I don't trust this approach to categorizing... there are too many exceptions. 

# class ReferenceAnnotator():
#     '''Assigns one of the following categories to a sequence based on the results of comparing its coordinates to the 
#     reference genome annotation. 
    
#     (1) match: Sequences where the boundaries exactly match the reference, or have 100 percent sequence identity with the reference. 
#     (2) conflict: Sequences which exceed max_overlap with any 
#     (3) intergenic
#     (4) pseudogene: A separate category is needed for pseudogenes, because it is hard to determine if it is a match or spurious
#         translation in these cases. This is the category for any predictions with same-strand overlap with a pseudogene.  
#     '''

#     categories = np.array(['match', 'intergenic', 'conflict', 'pseudogene'])

#     is_hypothetical = lambda df : df.top_hit_product == 'hypothetical protein'
#     is_ab_initio = lambda df : df.top_hit_evidence_type == 'ab initio prediction'
#     is_suspect = lambda df : ReferenceAnnotator.is_hypothetical(df) & ReferenceAnnotator.is_ab_initio(df) # This will be False for intergenic sequences. 

#     def __init__(self, max_overlap:int=50, min_sequence_identity:float=1):

#         self.max_overlap = max_overlap
#         self.min_sequence_identity = min_sequence_identity

#         self.is_pseudogene = lambda df : (df.same_strand & df.top_hit_pseudo)
#         self.is_match = lambda df : ~self.is_pseudogene(df) & df.in_frame & (df.top_hit_feature == 'CDS')
#         self.is_intergenic = lambda df :  ~self.is_match(df) & ~self.is_pseudogene(df) & (df.overlap_length < max_overlap) # Does not overlap with anything. 
#         self.is_conflict = lambda df : ~self.is_intergenic(df) & ~self.is_match(df) & ~self.is_pseudogene(df)  

#     @staticmethod
#     def _get_sequence_identity(seq:str, top_hitseq:str) -> float: 
#         aligner = PairwiseAligner(match_score=1, mismatch_score=0, gap_score=0)
#         alignment = aligner.align(seq, top_hitseq)[0] # I think this will get the best alignment?
#         score = alignment.score
#         score = max(score / len(seq), score / len(top_hitseq)) # Normalize the score by sequence length. 
#         return score
    
#     def _check(self, top_hits_df:pd.DataFrame):

#         top_hits_df['sequence_identity'] = np.where(top_hits_df.exact_match, 1, 0).astype(np.float64)
        
#         mask, n_downgraded = ((top_hits_df.category == 'match') & ~top_hits_df.exact_match), 0
#         for row in tqdm(top_hits_df[mask].itertuples(), total=mask.sum(), desc='ReferenceAnnotator._check'):
#             sequence_identity = ReferenceAnnotator._get_sequence_identity(row.query_seq, row.top_hit_seq)
#             top_hits_df.loc[row.Index, 'sequence_identity'] = sequence_identity
#             if (sequence_identity < self.min_sequence_identity):
#                 top_hits_df.loc[row.Index, 'category'] = 'conflict' if (row.overlap_length >= self.max_overlap) else 'intergenic'
#                 n_downgraded += 1

#         # I think there are some cases which are matches, but because one sequence is partial, they are not registering
#         # as in-frame. These are being categorized as conflicts or intergenic depending on the overlap. 

#         mask, n_upgraded = ((top_hits_df.category.isin(['intergenic', 'conflict'])) & top_hits_df.same_strand & (top_hits_df.top_hit_feature == 'CDS')), 0
#         for row in tqdm(top_hits_df[mask].itertuples(), total=mask.sum(), desc='ReferenceAnnotator._check'):
#             sequence_identity = ReferenceAnnotator._get_sequence_identity(row.query_seq, row.top_hit_seq)
#             top_hits_df.loc[row.Index, 'sequence_identity'] = sequence_identity
#             if (sequence_identity >= self.min_sequence_identity):
#                 top_hits_df.loc[row.Index, 'category'] = 'match'
#                 n_upgraded += 1
            
#         print(f'ReferenceAnnotator._check: Downgraded {n_downgraded} "match" sequences to "intergenic" or "conflict".')
#         print(f'ReferenceAnnotator._check: Upgraded {n_upgraded} "intergenic" or "conflict" sequences to "match.')

#         return top_hits_df 

#     def run(self, path:str):
#         top_hits_df = Reference.load(path)
#         conditions = [self.is_match(top_hits_df), self.is_intergenic(top_hits_df), self.is_conflict(top_hits_df), self.is_pseudogene(top_hits_df)]
#         top_hits_df['category'] = np.select(conditions, ReferenceAnnotator.categories, default='none')
#         assert (top_hits_df.category == 'none').sum() == 0, 'ReferenceAnnotator.run: Some sequences were not assigned annotations.'

#         top_hits_df['sequence_identity'] = 0.0
#         top_hits_df = self._check(top_hits_df)
#         top_hits_df.to_csv(path) # Write the DataFrame back to the original path. 
