import subprocess
import os 
from tqdm import tqdm


class AlphaFold():

    def __init__(self):
        pass 

    def get_structures(self, ids, dir_='../data/results/rpoz/pdbs'):
        version = 4 # Not sure if this version will always work. 
        for id_ in tqdm(ids, desc='get_structures'):
            url = f'https://alphafold.ebi.ac.uk/files/AF-{id_}-F1-model_v{version}.pdb'
            output_path = os.path.join(dir_, f'{id_}.pdb')
            if not os.path.exists(output_path):
                subprocess.run(f'wget {url} -O {output_path}', shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

