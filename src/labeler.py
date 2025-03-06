import pandas as pd 
import numpy as np 
from utils import * 
from src import get_genome_id, fillna
import os
from src.files import FASTAFile, InterProScanFile
import subprocess
from Bio.Align import PairwiseAligner


is_hypothetical = lambda df : df.top_hit_product == 'hypothetical protein'
is_ab_initio = lambda df : df.top_hit_evidence_type == 'ab initio prediction'
is_suspect = lambda df : is_hypothetical(df) & is_ab_initio(df) # This will be False for intergenic sequences. 


def get_alignment_scores(df:pd.DataFrame, seq_a_col:str='top_hit_seq', seq_b_col:str='query_seq', mode:str='local'): 
    aligner = PairwiseAligner(mode=mode, match_score=1, mismatch_score=0, gap_score=0)
    scores = list()
    seqs = list(zip(df[seq_a_col], df[seq_b_col]))
    for seq_a, seq_b in tqdm(seqs, desc='get_alignment_scores'):
        alignment = aligner.align(seq_a, seq_b)[0] # I think this will get the best alignment?
        score = alignment.score
        score = max(score / len(seq_a), score / len(seq_b)) # Normalize the score by sequence length. 
        scores.append(score)
    return np.array(scores)


class Labeler():
    categories = np.array(['match', 'intergenic', 'conflict'])
    interpro_cmd = '~/interproscan/interproscan-5.73-104.0/interproscan.sh'

    def __init__(self, path:str, max_overlap:int=50):
        self.genome_id = get_genome_id(path)

        is_match = lambda df : (df.in_frame & ~df.top_hit_pseudo) | (df.same_strand & df.top_hit_pseudo) 
        is_intergenic = lambda df : (df.overlap_length < max_overlap) & ~is_match(df) # Does not overlap with anything. 
        is_conflict = lambda df : ~is_intergenic(df) & ~is_match(df) # Seems to be in conflict with a real sequence. 

        self.labeled = dict()
        self.unlabeled = dict()

        self.needs_manual_validation = list()

        dtypes = {'top_hit_partial':str, 'query_partial':str, 'top_hit_translation_table':str, 'top_hit_codon_start':str}
        self.df = pd.read_csv(path, dtype=dtypes, index_col=0) # Load in the reference output. 
    
        mask = is_suspect(self.df)
        print(f'Labeler.__init__: {mask.sum()} out of {len(self.df)} sequences have suspect top hits and are being removed.')
        for row in self.df[~mask].itertuples():
            self._add_unlabeled(row.Index, category='none', reason='"suspect top hit"')
        self.df = self.df[~mask].copy()

        self.df['match'] = is_match(self.df)
        self.df['intergenic'] = is_intergenic(self.df)
        self.df['conflict'] = is_conflict(self.df)

        self.interpro_input_path = f'../data/interpro/{self.genome_id}_protein.faa'
        self.interpro_output_path = f'../data/interpro/{self.genome_id}_interpro.tsv'
        self._run_interpro(self.interpro_input_path, self.interpro_output_path)

    def _run_interpro(self, input_path:str, output_path:str):
        
        if not os.path.exists(output_path):
            mask = (self.df.intergenic) | (self.df.match & self.df.top_hit_pseudo) | self.df.conflict
            print(f'Labeler._run_interpro: Running InterProScan on {mask.sum()} sequences.')
            FASTAFile(df=self.df[mask].rename(columns={'query_seq':'seq'})).write(input_path)
            
            cmd = f'{Labeler.interpro_cmd} -i {input_path} -o {output_path} -f tsv'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _load_interpro(self, df:pd.DataFrame, max_e_value:float=None):
        interpro_cols = ['analysis', 'description', 'annotation', 'signature', 'signature_description'] # These are all string columns. 
        interpro_df = InterProScanFile(self.interpro_output_path).to_df(drop_duplicates=True, max_e_value=max_e_value)
        interpro_df = interpro_df[interpro_cols].rename(columns={col:f'interpro_{col}' for col in interpro_cols}).copy()

        df = df.merge(interpro_df, left_index=True, right_index=True, how='left', validate='one_to_one')
        df = fillna(df, rules={str:'none'})
        n_hits = (df.interpro_analysis != 'none').sum()
        # print(f'Labeler.add_interpro: Found InterProScan hits for {n_hits} out of {len(df)} sequences.')
        return df 

    def to_df(self, labeled_only:bool=False, unlabeled_only:bool=False):

        df = list()
        if (not unlabeled_only) and (len(self.labeled) > 0):
            df.append(pd.DataFrame(self.labeled))
        if (not labeled_only) and (len(self.unlabeled) > 0):
            df.append(pd.DataFrame(self.unlabeled))

        if len(df) > 0:
            df = pd.concat(df).T
            df.index.name = 'id'
            return df 

    def write(self, path:str, labeled_only:bool=False):
        df = self.to_df(labeled_only=labeled_only)
        df.to_csv(path) 

    def _add_labeled(self, id_:str, **row):
        row['pseudo'] = row.get('pseudo', False)
        row['auto'] = row.get('auto', True)
        self.labeled[id_] = row

    def _add_unlabeled(self, id_:str, **row):
        row['label'] = 'none'
        row['auto'] = row.get('auto', True)
        self.unlabeled[id_] = row

    def _label_match(self, df:pd.DataFrame, min_score:float=1):
        # Go ahead and add the exact matches to real to avoid computing alignments for everything. 
        mask = df.exact_match
        print(f'Labeler._label_match: {mask.sum()} out of {len(df)} sequences have exact boundary matches.')
        for row in df[mask].itertuples():
            reason = f'"exact match with {row.top_hit_protein_id}"' 
            self._add_labeled(row.Index, label='real', category='match', reason=reason)

        df = df[~mask].copy()
        df['score'] = np.round(get_alignment_scores(df), 2)
        mask = df.score >= min_score 
        print(f'Labeler._label_match: {mask.sum()} out of {len(df)} sequences meet the minimum alignment score threshold of {min_score}.')
        for row in df[mask].itertuples():
            reason = f'"alignment score {row.score} with {row.top_hit_protein_id}"' 
            self._add_labeled(row.Index, label='real', category='match', reason=reason)

        for row in df[~mask].itertuples():
            reason = f'"alignment with {row.top_hit_protein_id} (score={row.score}) did not meet threshold {min_score}"' 
            self._add_unlabeled(row.Index, category='match', reason=reason)

    def _label_interpro(self, df:pd.DataFrame, pseudo:bool=False, category:str=None):

        df = self._load_interpro(df)
        mask = (df.interpro_analysis == 'none')
        print(f'Labeler._label_interpro: {mask.sum()} out of {len(df)} {category} sequences have no InterProScan hit.')
        for row in df[mask].itertuples():
            self._add_unlabeled(row.Index, pseudo=pseudo, category=category, reason='"no InterProScan hit"')

        df = df[~mask].copy()
        mask = (df.interpro_analysis == 'AntiFam')
        print(f'Labeler._label_interpro: {mask.sum()} out of {len(df)} {category} sequences have an AntiFam InterProScan hit.')
        for row in df[mask].itertuples():
            self._add_labeled(row.Index, label='spurious', pseudo=pseudo, category=category, reason='"AntiFam InterProScan hit"')

        for row in df[~mask].itertuples():
            self.needs_manual_validation.append(row.Index)
            self._add_unlabeled(row.Index, pseudo=pseudo, category=category, reason='"needs manual validation"' )
    
    def _label_match_pseudo(self, df:pd.DataFrame):
        self._label_interpro(df, pseudo=True, category='match')
    
    def _label_intergenic(self, df:pd.DataFrame):
        self._label_interpro(df, category='intergenic')

    def _label_conflict(self, df:pd.DataFrame):
        self._label_interpro(df, category='conflict')

    def _label_auto(self, df:pd.DataFrame):

        self._label_match(df[df.match & ~df.top_hit_pseudo].copy())
        self._label_conflict(df[df.conflict].copy())
        self._label_match_pseudo(df[df.match & df.top_hit_pseudo].copy())
        self._label_intergenic(df[df.intergenic].copy())

        print(f'Labeler._label_auto: Automatically labeled {len(self.labeled)} sequences. {len(self.unlabeled)} could not be automatically-assigned labels.')

    def _label_manual(self, df:pd.DataFrame):

        def get_category(row):
            values = [getattr(row, category) for category in Labeler.categories]
            return Labeler.categories[values].item()

        def print_info(row):
            print()
            print(row.Index, get_category(row))
            print('query_length:', row.query_length, 'top_hit_length:', row.top_hit_length)
            print('overlap_length:', row.overlap_length, 'on the same strand' if row.same_strand else 'on opposite strands')
            print('top_hit_product:', row.top_hit_product, '(pseudogene)' if row.top_hit_pseudo else '')
            print('interpro:', row.interpro_analysis, row.interpro_signature, row.interpro_description, row.interpro_annotation)
            print('top_hit_note:', row.top_hit_note)
            print()

        df = self._load_interpro(df) # Everything marked for manual validation should have an InterProScan hit. 

        labeled = dict()
        for row in df.itertuples():
            print_info(row)
            category = get_category(row)
            label = input('label: ')
            reason = input('reason: ')

            if len(label) > 0 and len(reason) > 0:
                self._add_labeled(row.Index, category=category, auto=False, pseudo=row.top_hit_pseudo, reason=reason, label=label)
                del self.unlabeled[row.Index]

    def run(self, manual:bool=True, path:str=None):
        
        self._label_auto(self.df)
        self._label_manual(self.df[self.df.index.isin(self.needs_manual_validation)])



