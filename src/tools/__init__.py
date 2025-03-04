from src.tools.prodigal import Prodigal, Pyrodigal
from src.tools.ncbi import NCBIDatasets
from src.tools.mmseqs import MMseqs
from src.tools.uniref import UniRef
from src.files import GBFFFile, FASTAFile
from src import get_genome_id
import os 
import pandas as pd 
import numpy as np 

def download_homologs(path:str, pseudo_only:bool=True, output_dir='../data/proteins/homologs', overwrite:bool=False):
    '''Download homologous sequences used to infer protein existence, function, or coordinates in a GBFF file.'''
    output_path = os.path.join(output_dir, f'{get_genome_id(path)}_protein.faa')
    if os.path.exists(output_path) and (not overwrite):
        return 
        
    df = GBFFFile(path).to_df() # Load in the GBFF file from the specified path. 
    df = df[df.evidence_type == 'similar to AA sequence']
    df = df[df.pseudo] if pseudo_only else df

    tools = {'RefSeq':NCBIDatasets(), 'UniRef':UniRef()}

    protein_ids = {source:df.evidence_details for source, df_ in df.groupby('evidence_source')}
    print(f"download_homologs: Found homology information from sources: {', '.join(protein_ids.keys())}")

    homologs_df = []
    for source, protein_ids in protein_ids.items():
        if source not in tools:
            print(f'download_homologs: No tool to support downloading homologs from {source}.')
        tool = tools[source]
        df_ = tool.run(protein_ids=np.unique(protein_ids), path=None) # Expect tool to return a DataFrame with the index set as the protein ID. 
        tool.cleanup()
        homologs_df.append(df_)

    if len(homologs_df) == 0:
        print(f'download_homologs: No homologous proteins obtained for genome {get_genome_id(path)}.')
        return 
        
    homologs_df = pd.concat(homologs_df)
    FASTAFile(df=homologs_df).write(output_path)
    print(f'download_homologs: {len(homologs_df)} homologous sequences saved to {output_path}.')


genome_ids = ['GCF_025998455.1',
 'GCF_000014025.1',
 'GCF_001298465.1',
 'GCF_003660105.1',
 'GCF_008245025.1',
 'GCF_013372225.1',
 'GCF_004803815.1',
 'GCF_013372165.1',
 'GCF_002080395.1',
 'GCF_013372125.1',
 'GCF_003544835.1',
 'GCF_028751035.1',
 'GCF_002139935.1',
 'GCF_003544855.1',
 'GCF_013283835.1',
 'GCF_013201625.1',
 'GCF_001687475.2',
 'GCF_008245005.1',
 'GCF_000568815.1',
 'GCF_006459125.1',
 'GCF_000242915.1',
 'GCF_013201705.1',
 'GCF_004299785.2',
 'GCF_005083985.2',
 'GCF_013177655.1',
 'GCF_000736415.1',
 'GCF_013372205.1',
 'GCF_001602095.1',
 'GCF_013372105.1',
 'GCF_013201645.1',
 'GCF_013201605.1',
 'GCF_008245045.1',
 'GCF_003544935.1',
 'GCF_013372265.1',
 'GCF_001190745.1',
 'GCF_004803795.1',
 'GCF_000012965.1',
 'GCF_004214815.1',
 'GCF_005843985.1',
 'GCF_002104335.1',
 'GCF_900476215.1',
 'GCF_027946175.1',
 'GCF_000259275.1',
 'GCF_013201725.1',
 'GCF_013201895.1',
 'GCF_000092245.1',
 'GCF_001723605.1',
 'GCF_003346775.1',
 'GCF_000183725.1',
 'GCF_000265295.1',
 'GCF_000024885.1',
 'GCF_000147355.1',
 'GCF_900637325.1',
 'GCF_003355515.1',
 'GCF_002238335.1',
 'GCF_000816305.1',
 'GCF_000186245.1',
 'GCF_000017585.1',
 'GCF_000816185.1',
 'GCF_000200595.1',
 'GCF_000194135.1',
 'GCF_000021725.1',
 'GCF_013201665.1',
 'GCF_024267655.1',
 'GCF_013201825.1',
 'GCF_013177675.1',
 'GCF_008000835.1',
 'GCF_001956695.1',
 'GCF_003544915.1',
 'GCF_015265455.1',
 'GCF_017357825.1',
 'GCF_003346815.1',
 'GCF_004214795.1',
 'GCF_013201935.1',
 'GCF_003544815.1',
 'GCF_020911965.1',
 'GCF_013177635.1',
 'GCF_015265475.1',
 'GCF_009258225.1',
 'GCF_009068765.1',
 'GCF_014905115.1',
 'GCF_030296055.1',
 'GCF_016593255.1',
 'GCF_040436365.1',
 'GCF_014905095.1',
 'GCF_000987835.1',
 'GCF_014931715.1',
 'GCF_013201685.1',
 'GCF_014466985.1',
 'GCF_014905135.1',
 'GCF_001460635.1',
 'GCF_009258045.1',
 'GCF_015100395.1',
 'GCF_004803835.1',
 'GCF_000816785.1',
 'GCF_014217995.1',
 'GCF_027943725.1',
 'GCF_001693335.1',
 'GCF_000009085.1',
 'GCF_009730395.1',
 'GCF_011600945.2',
 'GCF_000284635.1',
 'GCF_013372245.1',
 'GCF_900638355.1',
 'GCF_032850805.1',
 'GCF_002139825.2',
 'GCF_013374215.1',
 'GCF_000007905.1',
 'GCF_002139855.1',
 'GCF_013416015.1',
 'GCF_002309535.1',
 'GCF_000009305.1',
 'GCF_003346755.1',
 'GCF_039555295.1',
 'GCF_003097575.1',
 'GCF_002021945.1',
 'GCF_018968685.1',
 'GCF_041735015.1',
 'GCF_900638485.1',
 'GCF_003711085.1',
 'GCF_000162575.1',
 'GCF_003667345.1',
 'GCF_001282945.1',
 'GCF_008693005.1',
 'GCF_002119425.1',
 'GCF_001283065.1',
 'GCF_000420385.1',
 'GCF_000507845.1',
 'GCF_000813325.1',
 'GCF_000597725.1',
 'GCF_900450885.1',
 'GCF_014253065.1',
 'GCF_905120475.1',
 'GCF_025770725.1',
 'GCF_002335445.1',
 'GCF_030347965.1',
 'GCF_030347995.1',
 'GCF_027859255.1',
 'GCF_018145655.1',
 'GCF_005771535.1',
 'GCF_004116335.1',
 'GCF_028649595.1',
 'GCF_021300615.1',
 'GCF_000763135.2',
 'GCF_005217605.1',
 'GCF_000744435.1',
 'GCF_000445475.1',
 'GCF_023646285.1',
 'GCF_003063295.1',
 'GCF_042649585.1',
 'GCF_001595645.1',
 'GCF_021300655.1',
 'GCF_019711685.1',
 'GCF_000687535.1',
 'GCF_900101285.1',
 'GCF_000518225.1',
 'GCF_030250085.1',
 'GCF_003364205.1',
 'GCF_003316695.1',
 'GCF_003063245.1',
 'GCF_003364265.1',
 'GCF_030686155.1',
 'GCF_905120465.1',
 'GCF_009192995.1',
 'GCF_016106035.1',
 'GCF_005406215.1',
 'GCF_005406225.1',
 'GCF_003670275.1',
 'GCF_000765905.2',
 'GCF_030249845.1',
 'GCF_003364255.1',
 'GCF_003670295.1',
 'GCF_003364195.1',
 'GCF_016937535.1',
 'GCF_009690845.1',
 'GCF_003660395.1',
 'GCF_003364335.1',
 'GCF_009208055.1',
 'GCF_003660285.1',
 'GCF_003245365.1',
 'GCF_006864425.1',
 'GCF_042649605.1',
 'GCF_000364285.1',
 'GCF_000813345.1',
 'GCF_900451005.1',
 'GCF_900196775.1',
 'GCF_003546685.2',
 'GCF_004116625.1',
 'GCF_003364315.1',
 'GCF_002138795.1',
 'GCF_020136095.1',
 'GCF_000765825.2',
 'GCF_900197855.1',
 'GCF_005406205.1',
 'GCF_007280785.1',
 'GCF_900198605.1',
 'GCF_900197775.1']

if __name__ == '__main__':
    from tqdm import tqdm
    for genome_id in tqdm(genome_ids):
        download_homologs(f'/home/prichter/Documents/tripy/data/proteins/ncbi/{genome_id}_genomic.gbff', pseudo_only=False, overwrite=True, output_dir='/home/prichter/Documents/tripy/data/proteins/homologs')
