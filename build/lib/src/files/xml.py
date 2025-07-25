import os 
import re 
from typing import List, Dict, Tuple 
from tqdm import tqdm 
from lxml import etree 
import pandas as pd 

# TODO: Read more about namespaces.

class XMLFile():

    databases = ['Pfam', 'InterPro', 'NCBIfam', 'PANTHER', 'SUPFAM', 'Gene3D', 'KEGG', 'GO', 'AntiFam']
    # features = ['region of interest', 'lipid moiety-binding region', 'binding site', 'transmembrane region', 'domain', 'compositionally biased region', 'active site']

    def find(self, elem, name:str, attrs:Dict[str, str]=None):
        '''Find the first tag in the entry element which has the specified names and attributes.'''
        xpath = f'.//{self.namespace}{name}' #TODO: Remind myself how these paths work. 
        if attrs is not None:
            for attr, value in attrs.items():
                xpath += f'[@{attr}=\'{value}\']'
        return elem.find(xpath)

    def findall(self, elem, name:str, attrs:Dict[str, str]=None):
        '''Find all tags in the entry element which have the specified names and attributes.'''
        xpath = f'.//{self.namespace}{name}'
        if attrs is not None:
            for attr, value in attrs.items():
                xpath += f'[@{attr}=\'{value}\']'
        return elem.findall(xpath)

    @staticmethod
    def get_tag(elem) -> str:
        namespace, tag = elem.tag.split('}') # Remove the namespace from the tag. 
        namespace = namespace + '}'
        return namespace, tag 
    
    def get_references(self, entry):
        references = dict()
        for database in XMLFile.databases:
            property_type = 'term' if (database == 'GO') else 'entry name' # Type of the tag to extract from the database entry. 
            database_entries = self.findall(entry, 'dbReference', attrs={'type':database})
            if len(database_entries) == 0:
                references[database.lower() + '_description'] = 'none'
                references[database.lower() + '_id'] = 'none'
            else:
                database_descriptions = [self.find(entry, 'property', attrs={'type':property_type}) for entry in database_entries]
                database_descriptions = [description.attrib['value'] if (description is not None) else 'no description' for description in database_descriptions]
                database_ids = [entry.attrib['id'] for entry in database_entries] # Every database reference should have an ID. 
                references[database.lower() + '_description'] = ';'.join(database_descriptions)
                references[database.lower() + '_id'] = ';'.join(database_ids)
        return references 
    
    def get_product(self, entry):
        product = {'product':'none'}
        # This should get the first fullName, which is the recommended name. 
        product['product'] = self.find(entry, 'fullName').text
        return product

    def get_function(self, entry) -> dict:
        function = {'function':'none'}
        function_entry = self.find(entry, 'comment', attrs={'type':'function'}) 
        if function_entry is not None: # Need to look at the "text" tag stored under the function entry.
           function['function'] = self.find(function_entry, 'text').text
        return function

    def get_taxonomy(self, entry) -> Dict[str, str]:
        '''Extract the taxonomy information from the organism tag group.'''
        lineage = [taxon.text for taxon in self.findall(entry, 'taxon')]
        taxonomy = dict()
        taxonomy['domain'] = lineage[0]
        taxonomy['lineage'] = ';'.join(lineage)
        taxonomy['species'] = self.find(entry, 'name').text
        taxonomy['ncbi_taxonomy_id'] = self.find(entry, 'dbReference', attrs={'type':'NCBI Taxonomy'}).attrib['id'] # , attrs={'type':'NCBI Taxonomy'})[0].id
        return taxonomy

    def get_refseq(self, entry) -> Dict[str, str]:
        '''Get references to RefSeq database in case I want to access the nucleotide sequence later on.'''
        refseq = dict()
        refseq_entry = self.find(entry, 'dbReference', attrs={'type':'RefSeq'}) # Can we assume there is always a RefSeq entry? No. 
        if (refseq_entry is not None):
            refseq['refseq_protein_id'] = refseq_entry.attrib['id']
            refseq['refseq_nucleotide_id'] = self.find(refseq_entry, 'property', attrs={'type':'nucleotide sequence ID'}).attrib['value']
        else:
            refseq['refseq_protein_id'] = 'none'
            refseq['refseq_nucleotide_id'] = 'none'
        return refseq

    def get_post_translational_modification(self, entry) -> Dict[str, str]:
        ptm = dict()
        ptm_entry = self.findall(entry, 'comment', attrs={'type':'PTM'})
        if (len(ptm_entry) > 0):
            text = [self.find(e, 'text').text for e in ptm_entry]
            text = ';'.join(text)
            ptm['post_translational_modification'] = text
        else:
            ptm['post_translational_modification'] = 'none'
        return ptm    

    def get_non_terminal_residue(self, entry) -> Dict[str, str]:
        '''If the entry passed into the function has a non-terminal residue(s), find the position(s) where it occurs; 
        there can be two non-terminal residues, one at the start of the sequence, and one at the end.'''
        # Figure out of the sequence is a fragment, i.e. if it has a non-terminal residue. 
        non_terminal_residue_entries = self.findall(entry, 'feature', attrs={'type':'non-terminal residue'})
        if len(non_terminal_residue_entries) > 0:
            positions = []
            for non_terminal_residue_entry in non_terminal_residue_entries:
                # Get the location of the non-terminal residue. 
                position = self.find(non_terminal_residue_entry, 'position').attrib['position']
                positions.append(position)
            positions = ','.join(positions)
        else:
            positions = 'none'
        return {'non_terminal_residue':positions}
                    
    def __init__(self, path:str, load_seqs:bool=True, chunk_size:int=100):
        self.path = path

        pbar = tqdm(etree.iterparse(path, events=('start', 'end')), desc=f'XMLFile.__init__: Parsing XML file.')
        entry, df = None, []
        for event, elem in pbar: # The file tree gets accumulated in the elem variable as the iterator progresses. 
            namespace, tag = XMLFile.get_tag(elem) # Extract the tag and namespace from the element. 
            self.namespace = namespace # Save the current namespace in the object.

            if (tag == 'entry') and (event == 'start'):
                entry = elem
            if (tag == 'entry') and (event == 'end'):
                accessions = [accession.text for accession in entry.findall(namespace + 'accession')]
                row = self.get_taxonomy(entry) 
                row.update(self.get_refseq(entry))
                row.update(self.get_non_terminal_residue(entry))
                row.update(self.get_function(entry))
                row.update(self.get_post_translational_modification(entry))
                row.update(self.get_references(entry))
                row.update(self.get_product(entry))

                # Why am I using findall here instead of just find? Maybe to get the latest version?
                row['seq'] = self.findall(entry, 'sequence')[-1].text
                # NOTE: It seems as though not all fragmented sequences are tagged with a fragment attribute.
                # row['fragment'] = 'fragment' in seq.attrib
                row['existence'] = self.find(entry, 'proteinExistence').attrib['type']
                row['name'] = self.find(entry, 'name').text 

                for accession in accessions:
                    row['id'] = accession 
                    df.append(row.copy())

                elem.clear() # Clear the element to avoid loading everything into memory. 
                pbar.update(len(accessions))
                pbar.set_description(f'XMLFile.__init__: Parsing XML file, row {len(df)}.')

        self.df = pd.DataFrame(df).set_index('id')

    def to_df(self):
        df = self.df.copy()
        df['file_name'] = os.path.basename(self.path)
        return df
    
    @staticmethod
    def get_features(path:str):
        
        features = set()
        feature_pattern = re.compile('<feature type="([^"]+)".{1,}description="([^"]+)".{1,}>')

        pbar = tqdm(etree.iterparse(path, events=('start', 'end')), desc=f'XMLFile.get_features: Parsing XML file.')

        f = open(path, 'r')
        for line in f:
            feature = re.search(feature_pattern, line)
            if (feature is not None) and (feature.group(1) != 'chain'):
                feature = (feature.group(1), feature.group(2))
                features.add(feature)
            pbar.update(1)
        f.close()
        pbar.close()
        return list(features)


