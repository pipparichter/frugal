import os 
import pandas as pd 
import subprocess
from tqdm import tqdm 
import shutil
import numpy as np 
from src.files import FASTAFile
import io 
from src import fillna
import requests 
import json
import re 

# TODO: Update AntiFam class to access sequences from InterPro instead of UniProt. 

# datasets summary genome taxon 2 --reference --annotated --assembly-level complete --mag exclude --assembly-source RefSeq --exclude-atypical --report sequence --as-json-lines | dataformat tsv genome-seq --fields accession,genbank-seq-acc,refseq-seq-acc,chr-name,seq-length,gc-percent

class NCBI():
    taxonomy_fields = ['Taxid', 'Tax name', 'Authority', 'Rank', 'Basionym', 'Basionym authority', 'Curator common name', 'Has type material', 'Group name', 'Superkingdom name', 'Superkingdom taxid', 'Kingdom name', 'Kingdom taxid', 'Phylum name', 'Phylum taxid', 'Class name', 'Class taxid', 'Order name', 'Order taxid', 'Family name', 'Family taxid', 'Genus name', 'Genus taxid', 'Species name', 'Species taxid'] 
    cleanup_files = ['README.md', 'md5sum.txt', 'ncbi.zip']
    cleanup_dirs = ['ncbi_dataset']

    src_dir = 'ncbi_dataset/data'
    src_file_names = {'gbff':'genomic.gbff', 'genome':'*genomic.fna', 'protein':'*protein.faa', 'gene':'gene.fna'}
    dst_file_names = {'gbff':'{genome_id}_genomic.gbff', 'genome':'{genome_id}_genomic.fna', 'protein':'{genome_id}_protein.faa'}

    def __init__(self):
        pass

    @staticmethod
    def _get_metadata(ids:list, cmd:str=None, path:str=None, chunk_size:int=20) -> pd.DataFrame:

        df = list()
        if (path is not None) and os.path.exists(path):
            df_ = pd.read_csv(path, sep='\t')
            print(f'NCBI._get_metadata: Found metadata entries for {len(df_)} IDs already in {path}.')
            ids = [id_ for id_ in ids if id_ not in df_.iloc[:, 0].values] # Don't repeatedly download the same ID.
            df.append(df_)

        n_chunks = 0 if (len(ids) == 0) else len(ids) // chunk_size + 1 # Handle case where ID list is empty.
        ids = [str(id_) for id_ in ids] # Convert the IDs to strings for joining, needed for the taxonomy IDs. 
        ids = [','.join(ids[i:i + chunk_size]) for i in range(0, n_chunks * chunk_size, chunk_size)]

        for id_ in tqdm(ids, desc='NCBI._get_metadata: Downloading metadata.'):
            try:
                output = subprocess.run(cmd.format(id_=id_), shell=True, check=True, capture_output=True)
                content = output.stdout.decode('utf-8').strip().split('\n')
                df_ = pd.read_csv(io.StringIO('\n'.join(content)), sep='\t')
                df_ = df_.drop(columns=['Query'], errors='ignore') # Drop the Query column, which is redundant. Only present when getting taxonomy metadata. 
                df.append(df_)
            except pd.errors.EmptyDataError as err: # Raised when the call to pd.read_csv fails, I think due to nothing written to stdout.
                print(f'NCBI._get_metadata: Failed on query {id_}. NCBI returned the following error message.')
                print(output.stderr.decode('utf-8'))
                return pd.concat(df) # Return everything that has been downloaded already to not lose progress.

        return pd.concat(df)

    def get_taxonomy_metadata(self, taxonomy_ids:list, path:str=None):

        cmd = 'datasets summary taxonomy taxon {id_} --as-json-lines | dataformat tsv taxonomy --template tax-summary'
        df = NCBI._get_metadata(taxonomy_ids, cmd=cmd, path=path)
        df = df.set_index('Taxid')

        if path is not None:
            df.to_csv(path, sep='\t')
        return df
    
    def get_genome_metadata(self, genome_ids:list, path:str='../data/ncbi_genome_metadata.tsv'):

        # https://www.ncbi.nlm.nih.gov/datasets/docs/v2/command-line-tools/using-dataformat/genome-data-reports/ 
        fields = 'accession,checkm-completeness,annotinfo-method,annotinfo-pipeline,assmstats-gc-percent,assmstats-total-sequence-len,'
        fields += 'assmstats-number-of-contigs,organism-tax-id,annotinfo-featcount-gene-pseudogene,annotinfo-featcount-gene-protein-coding,'
        fields += 'annotinfo-featcount-gene-non-coding'

        cmd = 'datasets summary genome accession {id_} --report genome --as-json-lines | dataformat tsv genome --fields ' + fields
        df = NCBI._get_metadata(genome_ids, cmd=cmd, path=path)
        df = fillna(df, rules={str:'none'}, check=False)
        df = df.set_index('Assembly Accession') 
        df.to_csv(path, sep='\t')
    
    def get_genomes(self, genome_ids:list, include:list=['gbff', 'genome'], dirs={'genome':'../data/ncbi/genomes', 'gbff':'../data/ncbi/gbffs'}):

        pbar = tqdm(genome_ids)
        for genome_id in pbar:
            pbar.set_description(f'NCBI.get_genomes: Downloading data for {genome_id}.')
            src_paths = [os.path.join(NCBI.src_dir, genome_id, NCBI.src_file_names[i]) for i in include]
            dst_paths = [os.path.join(dirs[i], NCBI.dst_file_names[i].format(genome_id=genome_id)) for i in include]

            if np.all([os.path.exists(path) for path in dst_paths]): # Skip if already downloaded. 
                continue

            cmd = f"datasets download genome accession {genome_id} --filename ncbi.zip --include {','.join(include)} --no-progressbar"
            try:
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
                # The -o option means that the ncbi.zip directory from the previous pass will be overwritten without prompting. 
                subprocess.run(f'unzip -o ncbi.zip -d .', shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Unpack the downloaded NCBI data package. 
                for src_path, dst_path in zip(src_paths, dst_paths):
                    subprocess.run(f'cp {src_path} {dst_path}', shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as err:
                print(f'NCBI.get_genomes: Failed to download data for {genome_id}. Returned error message "{err}"')

    def get_proteins(self, protein_ids:list, include=['protein'], path:str=None, chunk_size:int=20):
        # Need to break into chunks because the API doesn't support more than a handful of sequences. 
        n_chunks = np.ceil(len(protein_ids) / chunk_size).astype(int)
        protein_ids = [','.join(protein_ids[i:i + chunk_size]) for i in range(0, n_chunks * chunk_size, chunk_size)]
        
        df = list()
        src_path = os.path.join(NCBI.src_dir, 'protein.faa')
        for protein_ids_ in tqdm(protein_ids, 'NCBI.get_proteins'):
            cmd = f"datasets download gene accession {protein_ids_} --include protein --filename ncbi.zip"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # The -o option means that the ncbi.zip directory from the previous pass will be overwritten without prompting. 
            subprocess.run(f'unzip -o ncbi.zip -d .', shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            df.append(FASTAFile(path=src_path).to_df(prodigal_output=False))
        df = pd.concat(df)
        
        if path is not None:
            print(f'NCBI.get_proteins: Proteins saved to {path}')
            FASTAFile(df=df).write(path)
        return df 

    def cleanup(self):
        for file in NCBI.cleanup_files:
            if os.path.exists(file):
                os.remove(file)
        for dir_ in NCBI.cleanup_dirs:
            if os.path.isdir(dir_):
                shutil.rmtree(dir_)


class AntiFam():

    def __init__(self):
        pass 

    def get_antifams(self, path:str='../data/antifam.json'):
        antifams = []
        result = json.loads(requests.get('https://www.ebi.ac.uk/interpro/api/entry/antifam/').text)
        pbar = tqdm(total=result['count'], desc='AntiFam.get_antifams')
        while True: # Only returns 20 hits at a time, so need to paginate using the 'next' field. 
            antifams += [{'id':entry['metadata']['accession'], 'name':entry['metadata']['name']} for entry in result['results']]
            pbar.update(len(result['results']))
            if result['next'] is None:
                break
            result = json.loads(requests.get(result['next']).text)
        with open(path, 'w') as f:
            json.dump(antifams, f)
        print(f'AntiFam.get_antifams: AntiFam data written to {path}')

    @staticmethod
    def _get_taxonomy_metadata(taxonomy_ids:list):
        lineages = dict()
        for taxonomy_id in tqdm(taxonomy_ids, 'AntiFam._get_taxonomy_metadata'):
            try:
                url = f'https://www.ebi.ac.uk/interpro/api/taxonomy/uniprot/{taxonomy_id}?'
                result = json.loads(requests.get(url).text)
                lineages[taxonomy_id] = result['metadata']['lineage']
            except Exception as err:
                print(f'AntiFam._get_taxonomy_metadata: Failed on taxonomy id {taxonomy_id}. Returned error message "{err}"')
                lineages[taxonomy_id] = 'none'
        return lineages

    @staticmethod
    def _get_protein_info(entry:dict, antifam_id:str='none') -> dict:
        '''Extract information in a JSON entry for a protein.'''
        metadata = entry['metadata']
        info = dict()
        info['id'] = metadata['accession']
        info['antifam_id'] = antifam_id
        info['product'] = metadata['name']
        info['ncbi_taxonomy_id'] = metadata['source_organism']['taxId']
        info['organism'] = metadata['source_organism']['fullName']
        return info
    
    @staticmethod
    def _get_sequences(protein_ids:str) -> dict:
        failures = list()
        seqs = dict()
        for id_ in tqdm(protein_ids, desc='AntiFam._get_sequences.'):
            try:
                result = requests.get(f'https://rest.uniprot.org/uniprotkb/{id_}.fasta')
                seq = result.text.split('\n')[1:] # First entry is the latest version of the sequence. 
                seq = ''.join(seq)
                seqs[id_] = seq
            except:
                print(f'AntiFam._get_sequences: Failed on sequence {id_}.')
                failures.append(id_)
                seqs.append('none')
        return seqs 
    
    def get_proteins(self, antifam_ids:list, path:str=None, get_sequences:bool=True, get_taxonomy_metadata:bool=True) -> pd.DataFrame:

        df = []
        for id_ in antifam_ids:
            url = 'https://www.ebi.ac.uk/interpro/api/protein/unreviewed/entry/antifam/{antifam}?'.format(antifam=antifam['id'])
            result = json.loads(requests.get(url).text)
            pbar = tqdm(total=result['count'], desc='Downloading sequences for AntiFam family {antifam}.'.format(antifam=antifam['id']))
            while True:
                pbar.update(len(result['results']))
                df += [AntiFam._get_protein_info(entry, antifam_id=id_) for entry in result['results']]
                if result['next'] is None:
                    break
                result = json.loads(requests.get(result['next']).text)
            pbar.close()

        df = pd.DataFrame(df).set_index('id')
        df.to_csv(path)

        if get_sequences:
            seqs = AntiFam._get_sequences(list(df.index))
            df['seq'] = df.index.map(seqs)
            df.to_csv(path)
        
        if get_taxonomy_metadata:
            lineages = AntiFam._get_taxonomy_metadata(list(df.ncbi_taxonomy_id.unique()))
            df['lineage'] = df.ncbi_taxonomy_id.map(lineages)
            df.to_csv(path)
        
        print(f'AntiFam.get_proteins: AntiFam protein data written to {path}')
        return df 



# class UniRef():
#     url = 'https://rest.uniprot.org/uniref/stream?format=fasta&query=%28{protein_ids}%29'
#     def __init__(self):
#         pass

#     def get_proteins(self, protein_ids:list=None, path:str=None, chunk_size:int=20):
#         n_chunks = len(protein_ids) // chunk_size + 1
#         protein_ids = ['+OR+'.join(protein_ids[i:i + chunk_size]) for i in range(0, n_chunks * chunk_size, chunk_size)]
        
#         df = list()
#         for protein_ids_ in tqdm(protein_ids, desc='UniRef.run'):
#             url = UniRef.url.format(protein_ids=protein_ids)
#             cmd = f'wget "{url}"'

#             output = subprocess.run(cmd, shell=True, check=True, capture_output=True)
#             output = output.stderr.decode('utf-8')
#             src_path = re.search(r"Saving to: ([^\n]+)", output).group(1)[1:-1]

#             df.append(FASTAFile(src_path).to_df(prodigal_output=False))
#             os.remove(src_path)
#         df = pd.concat(df)

#         if path is not None:
#             print(f'UniRef.get_proteins: {len(df)} sequences saved to {path}')
#             FASTAFile(df=df).write(path)
#         return df 
    


# def fix_b_subtilis(database_path:str='../data/ncbi_cds.csv'):
#     genome_id = 'GCF_000009045.1' 

#     df = pd.read_csv(database_path, index_col=0, dtype={'partial':str}, low_memory=False)
#     bsub_df = df[df.genome_id == genome_id].copy()
#     df = df[df.genome_id != genome_id].copy()

#     evidence_types = []
#     for row in bsub_df.itertuples():
#         if ('Evidence 1' in row.note) or ('Evidence 2' in row.note):
#             evidence_types.append('experiment')
#         elif ('Evidence 4' in row.note) or ('Evidence 3' in row.note):
#             evidence_types.append('similar to sequence')
#         else:
#             evidence_types.append('ab initio prediction')

#     bsub_df['evidence_type'] = evidence_types
#     df = pd.concat([df, bsub_df])
#     print(f'fix_b_subtilis: Writing modified database to {database_path}')
#     df.to_csv(database_path)


