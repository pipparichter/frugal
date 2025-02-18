from src import seed 
seed(42) # Make sure everything is random-seeded for reproducibility. 

import numpy as np 
import pandas as pd 
from src.build import Build 
from src.dataset import Dataset, split, Datasets
from src.sampler import Sampler
from src.classifier import Classifier
import argparse
from src.genome import ReferenceGenome
from src.files import FASTAFile
from src.embedders import get_embedder
import re
import glob
from tqdm import tqdm 
import os
from src import get_genome_id

        

def build():
    pass 


def ref():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', nargs='+', type=str)
    parser.add_argument('--output-dir', default='../data/ref.out/', type=str)
    parser.add_argument('--ref-dir', default='../data/refseq/', type=str)
    parser.add_argument('--prodigal-output', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # pbar = tqdm(glob.glob(args.input-path), desc='ref: Searching reference genomes.')
    pbar = tqdm(args.input_path, desc='ref: Searching reference genomes.')

    for path in pbar:

        genome_id = get_genome_id(path)
        assert genome_id is not None, f'ref: Could not extract a genome ID from the input path {path}.'

        output_path = os.path.join(args.output_dir, f'{genome_id}.ref.csv')
        pbar.set_description(f'ref: Searching reference genome for {genome_id}.')
        
        if os.path.exists(output_path) and (not args.overwrite):
            continue

        ref_path = os.path.join(args.ref_dir, f'{genome_id}_genomic.gbff')
        genome = ReferenceGenome(ref_path, genome_id=genome_id)

        df = FASTAFile(path).to_df(prodigal_output=args.prodigal_output)
        results_df = genome.search(df, verbose=False)

        results_df.to_csv(output_path)

    print(f'ref: Search complete. Output written to {args.output_dir}')


# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 300GB --partition gpu --gres gpu:1 --time 24:00:00 --wrap "embed -i ../data/filter_dataset_train.csv"
def embed():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', default=None, type=str)
    parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    output_path = args.output_path if (args.output_path is not None) else args.input_path.replace('.csv', '.h5')

    # Sort the sequences in order of length, so that the longest sequences are first. This is so that the 
    # embedder works more efficiently, and if it fails, it fails early in the embedding process.
    df = pd.read_csv(args.input_path, index_col=0)
    df = df.iloc[np.argsort(df.seq.apply(len))[::-1]]

    store = pd.HDFStore(output_path, mode='a' if (not args.overwrite) else 'w', table=True) # Should confirm that the file already exists. 
    existing_keys = [key.replace('/', '') for key in store.keys()]

    if 'metadata' in existing_keys:
        df_ = store.get('metadata')
        assert np.all(df.index == store.get('metadata')), 'embed: The input metadata and existing metadata do not match.'
    else:
        store.put('metadata', df, format='table', data_columns=None)

    if (args.feature_type not in existing_keys) or (overwrite):
        print(f'embed: Generating embeddings for {args.feature_type}.')
        embedder = get_embedder(args.feature_type)
        embeddings = embedder(df.seq.values.tolist())
        store.put(args.feature_type, pd.DataFrame(embeddings, index=df.index), format='table', data_columns=None) 

    store.close()


# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 300GB --partition gpu --gres gpu:1 --time 24:00:00 --wrap "train --input-path ../data/filter_dataset_train.csv --model-name test"
def train():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--output-dir', default='./models', type=str)
    parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    parser.add_argument('--balance-classes', action='store_true')
    parser.add_argument('--balance-lengths', action='store_true')
    parser.add_argument('--weight-loss', action='store_true')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=16, type=int)

    args = parser.parse_args()

    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, load_seqs=True, load_labels=True)
    model = Classifier(dims=(dataset.n_features, 512, dataset.n_classes))
    dataset_train, dataset_test = split(dataset)
    model.scale(dataset_train, fit=True)
    model.scale(dataset_test, fit=False)

    sampler, batch_size = None, args.batch_size
    if (args.balance_classes or args.balance_lengths):
        sampler = Sampler(dataset_train, batch_size=args.batch_size, balance_classes=args.balance_classes, balance_lengths=args.balance_lengths)
        batch_size = None 
    
    model.fit(Datasets(dataset_train, dataset_test), batch_size=batch_size, sampler=sampler, epochs=args.epochs, weight_loss=args.weight_loss)
    output_path = os.path.join(args.output_dir, args.model_name + '.pkl')
    model.save(path)
    print(f'train: Saved trained model to {output_path}')