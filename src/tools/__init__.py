from src.tools.prodigal import Prodigal, Pyrodigal
from src.tools.mmseqs import MMSeqs

# def download_homologs(path:str, pseudo_only:bool=True, output_dir='../data/proteins/homologs', overwrite:bool=False):
#     '''Download homologous sequences used to infer protein existence, function, or coordinates in a GBFF file.'''
#     output_path = os.path.join(output_dir, f'{get_genome_id(path)}_protein.faa')
#     if os.path.exists(output_path) and (not overwrite):
#         return 
        
#     df = GBFFFile(path).to_df() # Load in the GBFF file from the specified path. 
#     df = df[df.evidence_type == 'similar to AA sequence']
#     df = df[df.pseudo] if pseudo_only else df

#     tools = {'RefSeq':NCBIDatasets(), 'UniRef':UniRef()}

#     protein_ids = {source:df.evidence_details for source, df_ in df.groupby('evidence_source')}
#     print(f"download_homologs: Found homology information from sources: {', '.join(protein_ids.keys())}")

#     homologs_df = []
#     for source, protein_ids in protein_ids.items():
#         if source not in tools:
#             print(f'download_homologs: No tool to support downloading homologs from {source}.')
#         tool = tools[source]
#         df_ = tool.run(protein_ids=np.unique(protein_ids), path=None) # Expect tool to return a DataFrame with the index set as the protein ID. 
#         tool.cleanup()
#         homologs_df.append(df_)

#     if len(homologs_df) == 0:
#         print(f'download_homologs: No homologous proteins obtained for genome {get_genome_id(path)}.')
#         return 
        
#     homologs_df = pd.concat(homologs_df)
#     FASTAFile(df=homologs_df).write(output_path)
#     print(f'download_homologs: {len(homologs_df)} homologous sequences saved to {output_path}.')
