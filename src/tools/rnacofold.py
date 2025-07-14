import os 
import subprocess
from src.files import FASTAFile
import pandas as pd
import io

anti_sd_seq = 'ACCUCCUUA'

class RNACoFold():

    cmd = 'RNAcofold'

    def __init__(self):

        self.input_path = None
        self.output_paths = list()

    def _make_input_file(self, df:pd.DataFrame, ref_seq:str=anti_sd_seq, input_path:str='tmp.fna', seq_col:str='seq'):
        
        df = df.copy() # Otherwise the seq column in the input gets overwritten. 
        df['seq'] = [f'{seq}&{ref_seq}' for seq in df[seq_col]]
        file = FASTAFile(df=df)
        file.write(input_path)
        return input_path
    
    def cleanup(self):

        for path in [self.input_path] + self.output_paths:
            if os.path.exists(path):
                os.remove(path)

    def run(self, df:pd.DataFrame, seq_col:str='seq', output_path:str=None):

        self.output_paths = [f'{id_}_dp.ps' for id_ in df.index]
        self.output_paths = [f'{id_}_ss.ps' for id_ in df.index]
        
        self.input_path = self._make_input_file(df, seq_col=seq_col)
        csv = subprocess.run(f'{RNACoFold.cmd} --output-format D {self.input_path}', shell=True, check=True, capture_output=True, text=True).stdout
        self.cleanup()

        output_df = pd.read_csv(io.StringIO(csv), sep=',')
        output_df = output_df.set_index('seq_id')
        output_df.index.name = 'id'

        if output_path is not None:
            output_df.to_csv(output_path)

        return output_df
        # −−output−format D