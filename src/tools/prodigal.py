import os 
import subprocess 


def get_output_path(input_path:str, output_dir:str):
    input_file_name = os.path.basename(input_path)
    output_file_name = input_file_name.replace('_genomic.fna', '_protein.faa')
    return os.path.join(output_dir, output_file_name)


class Pyrodigal():

    def __init__(self, output_dir:str='../data/prodigal'):
        self.output_dir = output_dir 

    def run(self, input_path:str, output_path:str=None, min_length:int=10, max_overlap:int=None, translation_table:int=11):
        if max_overlap is None: # Maximum allowed overlap must be less than the minimum gene length. 
            max_overlap = min_length - 1

        if output_path is None:
            output_path = get_output_path(input_path, self.output_dir)

        cmd = f'pyrodigal -i {input_path} -a {output_path} --max-overlap {max_overlap} --min-gene {min_length} -g {translation_table}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Prodigal():

    def __init__(self, output_dir:str='../data/prodigal'):
        self.output_dir = output_dir 

    def run(self, input_path:str, output_path:str=None, translation_table:int=11):
        
        if output_path is None:
            output_path = get_output_path(input_path, self.output_dir)

        cmd = f'prodigal -i {input_path} -a {output_path} -g {translation_table}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    