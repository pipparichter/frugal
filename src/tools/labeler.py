import pandas as pd 
import numpy as np 
from src import get_genome_id, fillna
import os
from src.files import FASTAFile, InterProScanFile
import subprocess
from Bio.Align import PairwiseAligner
import glob 


is_hypothetical = lambda df : df.top_hit_product == 'hypothetical protein'
is_ab_initio = lambda df : df.top_hit_evidence_type == 'ab initio prediction'
is_suspect = lambda df : is_hypothetical(df) & is_ab_initio(df) # This will be False for intergenic sequences. 


def get_alignment_scores(df:pd.DataFrame, seq_a_col:str='top_hit_seq', seq_b_col:str='query_seq', mode:str='local'): 
    aligner = PairwiseAligner(mode=mode, match_score=1, mismatch_score=0, gap_score=0)
    scores = list()
    for seq_a, seq_b in zip(df[seq_a_col], df[seq_b_col]):
        alignment = aligner.align(seq_a, seq_b)[0] # I think this will get the best alignment?
        score = alignment.score
        score = max(score / len(seq_a), score / len(seq_b)) # Normalize the score by sequence length. 
        scores.append(score)
    return np.array(scores)


class Labeler():
    categories = np.array(['match', 'intergenic', 'conflict'])
    labels = ['real', 'spurious']
    interpro_cmd = '~/interproscan/interproscan-5.73-104.0/interproscan.sh'
    interpro_cols = ['analysis', 'description', 'annotation', 'signature', 'signature_description'] # These are all string columns. 

    def __init__(self, path:str, max_overlap:int=50, interpro_dir:str='../data/interpro'):
        self.genome_id = get_genome_id(path)

        self.is_match = lambda df : (df.in_frame & ~df.top_hit_pseudo) | (df.same_strand & df.top_hit_pseudo) 
        self.is_intergenic = lambda df : (df.overlap_length < max_overlap) & ~is_match(df) # Does not overlap with anything. 
        self.is_conflict = lambda df : ~is_intergenic(df) & ~is_match(df) # Seems to be in conflict with a real sequence. 

        dtypes = {'top_hit_partial':str, 'query_partial':str, 'top_hit_translation_table':str, 'top_hit_codon_start':str}
        self.df = pd.read_csv(path, dtype=dtypes, index_col=0) # Load in the reference output. 

        self.interpro_input_path = os.path.join(interpro_dir, f'{self.genome_id}_protein.faa')
        self.interpro_output_path = os.path.join(interpro_dir, f'{self.genome_id}_interpro.tsv')

        self.labeled = dict()
        self.unlabeled = dict()
        self.needs_manual_validation = list()

        self.has_manual_labels = False

    def _run_interpro(self, input_path:str, output_path:str):
        
        if not os.path.exists(output_path):
            mask = (self.df.intergenic) | (self.df.match & self.df.top_hit_pseudo) | self.df.conflict
            print(f'Labeler._run_interpro: Running InterProScan on {mask.sum()} sequences.')
            FASTAFile(df=self.df[mask].rename(columns={'query_seq':'seq'})).write(input_path)
            
            cmd = f'{Labeler.interpro_cmd} -i {input_path} -o {output_path} -f tsv'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _load_interpro(self, df:pd.DataFrame, max_e_value:float=None):
        interpro_df = InterProScanFile(self.interpro_output_path).to_df(drop_duplicates=True, max_e_value=max_e_value)
        interpro_df = interpro_df[Labeler.interpro_cols].rename(columns={col:f'interpro_{col}' for col in Labeler.interpro_cols}).copy()

        n = len(df)
        df = df.merge(interpro_df, left_index=True, right_index=True, how='left', validate='one_to_one')
        assert len(df) == n, f'Labeler._load_interpro: Expected {n} entries after merging InterProScan results, but {len(df)} are present.'

        df = fillna(df, rules={str:'none'})
        n_hits = (df.interpro_analysis != 'none').sum()
        # print(f'Labeler.add_interpro: Found InterProScan hits for {n_hits} out of {len(df)} sequences.')
        return df 

    def to_df(self):
        df = list()
        if len(self.labeled) > 0:
            df.append(pd.DataFrame(self.labeled).T)
        if len(self.unlabeled) > 0:
            df.append(pd.DataFrame(self.unlabeled).T)
        if len(df) > 0:
            df = pd.concat(df, ignore_index=False)
            df.index.name = 'id'
            return df 

    @staticmethod
    def get_manual_validation_candidates(df:pd.DataFrame):
        mask = (df.category == 'match') & (df.reason == 'needs manual validation')
        mask = mask | ((df.category == 'conflict') & df.reason == 'no InterProScan hit')
        return list(df[mask].index)

    @classmethod    
    def load(cls, labels_path:str=None, ref_path:str=None, interpro_dir:str='../data/interpro', max_overlap:int=50):

        labeler = cls(path=ref_path, interpro_dir=interpro_dir, max_overlap=max_overlap)

        df = pd.read_csv(labels_path, index_col=0)

        labeler.needs_manual_validation = Labeler.get_manual_validation_candidates(df)
        labeler.unlabeled = df[df.label == 'none'].to_dict(orient='index')
        labeler.unlabeled = df[df.label != 'none'].to_dict(orient='index')
        labeler.has_manual_labels = (~df.auto).sum() > 0 # Check if all labels are auto or not. 

        return labeler

    def write(self, path:str):
        df = self.to_df()
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
            reason = f'exact match with {row.top_hit_protein_id}' 
            self._add_labeled(row.Index, label='real', category='match', reason=reason)

        df = df[~mask].copy()
        df['score'] = np.round(get_alignment_scores(df), 2)
        mask = df.score >= min_score 
        print(f'Labeler._label_match: {mask.sum()} out of {len(df)} sequences meet the minimum alignment score threshold of {min_score}.')
        for row in df[mask].itertuples():
            reason = f'alignment score {row.score} with {row.top_hit_protein_id}' 
            self._add_labeled(row.Index, label='real', category='match', reason=reason)

        for row in df[~mask].itertuples():
            reason = f'alignment with {row.top_hit_protein_id} (score={row.score}) did not meet threshold {min_score}' 
            self._add_unlabeled(row.Index, category='match', reason=reason)

    def _label_interpro(self, df:pd.DataFrame, pseudo:bool=False, category:str=None):

        df = self._load_interpro(df)
        mask = (df.interpro_analysis == 'none')
        print(f'Labeler._label_interpro: {mask.sum()} out of {len(df)} {category} sequences have no InterProScan hit.')
        for row in df[mask].itertuples():
            self._add_unlabeled(row.Index, pseudo=pseudo, category=category, reason='no InterProScan hit')

        df = df[~mask].copy()
        mask = (df.interpro_analysis == 'AntiFam')
        print(f'Labeler._label_interpro: {mask.sum()} out of {len(df)} {category} sequences have an AntiFam InterProScan hit.')
        for row in df[mask].itertuples():
            self._add_labeled(row.Index, label='spurious', pseudo=pseudo, category=category, reason='AntiFam InterProScan hit')

        for row in df[~mask].itertuples():
            self.needs_manual_validation.append(row.Index)
            self._add_unlabeled(row.Index, pseudo=pseudo, category=category, reason='needs manual validation' )
    
    def _label_match_pseudo(self, df:pd.DataFrame):
        self._label_interpro(df, pseudo=True, category='match')
    
    def _label_intergenic(self, df:pd.DataFrame):
        self._label_interpro(df, category='intergenic')

    def _label_conflict(self, df:pd.DataFrame):
        self._label_interpro(df, category='conflict')

    def _label_auto(self, df:pd.DataFrame):

        mask = is_suspect(df)
        print(f'Labeler.__init__: {mask.sum()} out of {len(self.df)} sequences have suspect top hits and are being removed.')
        for row in self.df[mask].itertuples():
            self._add_unlabeled(row.Index, category='none', pseudo=row.top_hit_pseudo, reason='suspect top hit')
        df = df[~mask].copy()

        self.df['match'] = self.is_match(self.df)
        self.df['intergenic'] = self.is_intergenic(self.df)
        self.df['conflict'] = self.is_conflict(self.df)

        self._run_interpro(self.interpro_input_path, self.interpro_output_path)

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
            print('query_length:', row.query_length, 'top_hit_length:', row.top_hit_length)
            print('overlap_length:', row.overlap_length, 'on the same strand' if row.same_strand else 'on opposite strands')
            print('top_hit_product:', row.top_hit_product, '(pseudogene)' if row.top_hit_pseudo else '')
            print('interpro:', ' '.join([getattr(row, f'interpro_{col}') for col in Labeler.interpro_cols]))
            print('top_hit_note:', row.top_hit_note)

        df = self._load_interpro(df) # Everything marked for manual validation should have an InterProScan hit. 

        labeled = dict()
        n = len(df)
        for i, row in enumerate(df.itertuples()):
            category = get_category(row)
            print(f'\n{i}/{n} {row.Index} ({category})')
            print_info(row)

            label = None
            while label not in Labeler.labels + ['none']:
                label = input('label: ')
            reason = input('reason: ')

            if (label in Labeler.labels):
                self._add_labeled(row.Index, category=category, auto=False, pseudo=row.top_hit_pseudo, reason=f'{reason}', label=label)
                del self.unlabeled[row.Index]

    def run(self, add_auto_labels:bool=True, add_manual_labels:bool=True):
        if add_auto_labels: 
            self._label_auto(self.df)
        if add_manual_labels:
            print('Labeler.run: Entering manual mode.')
            self._label_manual(self.df[self.df.index.isin(self.needs_manual_validation)])



# I think the approach I will take is to be more strict about what I am considering real, and also handle pseudogenes separately. 

# To consider a sequence to be a true match, I will look at Prodigal sequences which are in-frame with non-pseudo reference sequences. 
# I don't want to require similar lengths (because this will miss the selenoproteins), but should perhaps check by comparing sequence
# identities. I will not consider any matches which are matches to suspect sequences. 

# For the pseudogenes, I will need to consider both in-frame and not in-frame hits. I should check, but I am assuming that every Prodigal-
# predicted sequence that is a translated component of a pseudogene should have query_overlap_fraction of 1. Also, will still need to require
# them being on the same strand. As a final check, I will run InterProScan

# I will run InterProScan on all intergenic sequences, and take those with AntiFam annotations to be spurious. I am hesitant to classify anything
# with no InterPro label as non-spurious, as I don't fully trust InterPro to capture everything. 

# Sequences that conflict with non-pseudo coding sequences and RNA genes are also potentially spurious. I will also run InterPro on the conflicts, and
# take anything which cannot be assigned an annotation, or is annotated as AntiFam. 

# Ultimately, going to want to run InterPro analysis on (1) potential pseudogene matches, (2) intergenic sequences, (3), conflicts. 

# I think I might want to write code to manually validate the pseudogene hits. 

# Maybe I should include the conflicting sequences with hypothetical sequences in the InterPro analysis?