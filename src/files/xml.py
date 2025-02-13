import os 
import re 
from typing import List, Dict, Tuple 
from tqdm import tqdm 
from lxml import etree 

# TODO: Read more about namespaces

class XMLFile():

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
        # Namespaces look like [EXAMPLE] specify the location in the tree. 
        namespace, tag = elem.tag.split('}') # Remove the namespace from the tag. 
        namespace = namespace + '}'
        return namespace, tag 

    def get_annotation(self, entry) -> Dict[str, str]:
        '''Grab the functional description and KEGG ortho group (if they exist) for the entry.'''
        annotation = dict()
        kegg_entry = self.find(entry, 'dbReference', attrs={'type':'KEGG'}) 
        if kegg_entry is not None:
            annotation['kegg'] = kegg_entry.attrib['id']
        function_entry = self.find(entry, 'comment', attrs={'type':'function'}) 
        if function_entry is not None:
            # Need to look at the "text" tag stored under the function entry.
            annotation['function'] = self.find(function_entry, 'text').text 
        return annotation

    def get_taxonomy(self, entry) -> Dict[str, str]:
        '''Extract the taxonomy information from the organism tag group.'''
        levels = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        taxonomy = {level:taxon.text for taxon, level in zip(self.findall(entry, 'taxon'), levels)}
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
            refseq['refseq_protein_id'] = None
            refseq['refseq_nucleotide_id'] = None
        return refseq

    def get_non_terminal_residue(self, entry) -> Dict[str, str]:
        '''If the entry passed into the function has a non-terminal residue(s), find the position(s) where it occurs; 
        there can be two non-terminal residues, one at the start of the sequence, and one at the end.'''
        # Figure out of the sequence is a fragment, i.e. if it has a non-terminal residue. 
        non_terminal_residue_entries = self.findall(entry, 'feature', attrs={'type':'non-terminal residue'})
        # assert len(non_terminal_residues) < 2, f'XMLFile.__init__: Found more than one ({len(non_terminal_residue)}) non-terminal residue, which is unexpected.'
        if len(non_terminal_residue_entries) > 0:
            positions = []
            for non_terminal_residue_entry in non_terminal_residue_entries:
                # Get the location of the non-terminal residue. 
                position = self.find(non_terminal_residue_entry, 'position').attrib['position']
                positions.append(position)
            positions = ','.join(positions)
        else:
            positions = None
        return {'non_terminal_residue':positions}
                    
    def __init__(self, path:str, load_seqs:bool=True, chunk_size:int=100):
        super().__init__(path)

        pbar = tqdm(etree.iterparse(path, events=('start', 'end')), desc=f'XMLFile.__init__: Parsing XML file {self.file_name}...')
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
                row.update(self.get_annotation(entry))

                if load_seqs:
                    # Why am I using findall here instead of just find?
                    seq = self.findall(entry, 'sequence')[-1]
                    row['seq'] = seq.text
                    # NOTE: It seems as though not all fragmented sequences are tagged with a fragment attribute.
                    # row['fragment'] = 'fragment' in seq.attrib
                row['existence'] = self.find(entry, 'proteinExistence').attrib['type']
                row['name'] = self.find(entry, 'name').text 

                for accession in accessions:
                    row['id'] = accession 
                    df.append(row.copy())

                elem.clear() # Clear the element to avoid loading everything into memory. 
                pbar.update(len(accessions))
                pbar.set_description(f'XMLFile.__init__: Parsing NCBI XML file, row {len(df)}...')

        self.df = pd.DataFrame(df).set_index('id')


    def to_df(self):
        df = self.df.copy()
        df['file_name'] = self.file_name
        return df