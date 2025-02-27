from src import seed 
seed(42) # Make sure everything is random-seeded for reproducibility. 

import numpy as np 
import copy 
import pandas as pd 
from src.dataset import Dataset, split, Datasets
from src.sampler import Sampler
from src.classifier import Classifier
import argparse
from src.genome import ReferenceGenome
from src.files import FASTAFile
from src.embed import get_embedder
from src.embed.library import EmbeddingLibrary
import re
import glob
from tqdm import tqdm 
import os
from src import get_genome_id, fillna
from multiprocessing import Pool
import src.embed.library
from transformers import logging
logging.set_verbosity_error() # Turn off the warning about uninitialized weights. 


def build():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title='build', dest='subcommand', required=True)

    parser_library = subparser.add_parser('library')
    parser_library.add_argument('--feature-type', default='esm_650m_gap', type=str)
    parser_library.add_argument('--input-dir', type=str, default='./data/proteins/prodigal')
    parser_library.add_argument('--library-dir', type=str, default='./data/embeddings')
    # parser_library.add_argument('--max-length', type=int, default=2000)
    parser_library.add_argument('--parallelize', action='store_true')
    parser_library.add_argument('--n-processes', default=4, type=int)

    args = parser.parse_args()
    
    if args.subcommand == 'library':
        build_library(args)


# sbatch --mail-user prichter@caltech.edu --mail-type ALL --partition gpu --gpus-per-task 4 --mem-per-gpu 300GB --gres gpu --time 24:00:00 --wrap "build library"
def build_library(args):

    lib = EmbeddingLibrary(dir_=args.library_dir, feature_type=args.feature_type)
    file_names = os.listdir(args.input_dir)

    if args.parallelize:
        n_processes = 5
        inputs = [[copy.copy(lib)] + list(file_names) for file_names in np.array_split(file_names, len(file_names) // 10)]
        pool = Pool(args.n_processes)
        pool.starmap(src.embed.library.add, inputs)
        pool.close()
    else:
        src.embed.library.add(*file_names)


def ref():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', nargs='+', type=str)
    parser.add_argument('--output-dir', default='../data/ref/', type=str)
    parser.add_argument('--database-path', default='../data/ncbi_cds.csv', type=str)
    parser.add_argument('--prodigal-output', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    pbar = tqdm(args.input_path)
    for path in pbar:
        genome_id = get_genome_id(path, errors='raise')
        results_output_path = os.path.join(args.output_dir, f'{genome_id}_results.csv')
        summary_output_path = os.path.join(args.output_dir, f'{genome_id}_summary.csv')

        pbar.set_description(f'ref: Searching reference for {genome_id}.')
        if os.path.exists(results_output_path) and (not args.overwrite):
            print(f'ref: Search results for {genome_id} are already present in {args.output_dir}.')
            continue

        genome = ReferenceGenome(path)
        query_df = FASTAFile(path=path).to_df(prodigal_output=args.prodigal_output)
        results_df, summary_df = genome.search(query_df, verbose=False, summarize=args.summarize)

        results_df.to_csv(results_output_path)
        if not (summary_df is None):
            summary_df.to_csv(summary_output_path)

    print(f'ref: Search complete. Results written to {args.output_dir}')


# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 500GB --partition gpu --gres gpu:1 --time 24:00:00 --wrap "embed --input-path ./data/filter_dataset_train.csv"
def embed():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', default=None, type=str)
    parser.add_argument('--feature-type', nargs='+', default=['esm_650m_gap', 'esm_3b_gap', 'pt5_3b_gap'], type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--library-dir', default='../data/embeddings')
    args = parser.parse_args()

    output_path = args.output_path if (args.output_path is not None) else args.input_path.replace('.csv', '.h5')

    # Sort the sequences in order of length, so that the longest sequences are first. This is so that the 
    # embedder works more efficiently, and if it fails, it fails early in the embedding process.
    df = pd.read_csv(args.input_path, index_col=0)
    df = df.iloc[np.argsort(df.seq.apply(len))[::-1]]

    store = pd.HDFStore(output_path, mode='a' if (not args.overwrite) else 'w', table=True) # Should confirm that the file already exists. 
    existing_keys = [key.replace('/', '') for key in store.keys()]

    if 'metadata' in existing_keys:
        df_ = store.get('metadata') # Load existing metadata. 
        assert np.all(df.index == df_.index), 'embed: The input metadata and existing metadata do not match.'
    else: # Storing in table format means I can add to the file later on. 
        df = fillna(df, rules={str:'none', bool:False}, check=False)
        store.put('metadata', df, format='table', data_columns=None)

    for feature_type in args.feature_type:
        if (feature_type not in existing_keys) or (overwrite):
            print(f'embed: Generating embeddings for {feature_type}.')
            embedder = get_embedder(feature_type)
            embeddings = embedder(df.seq.values.tolist())
            store.put(feature_type, pd.DataFrame(embeddings, index=df.index), format='table', data_columns=None) 

    store.close()


# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 300GB --partition gpu --gres gpu:1 --time 24:00:00 --wrap "train --input-path ./data/filter_dataset_train.h5 --balance-classes --model-name filter_esm_650m_gap_v1"
# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 300GB --partition gpu --gres gpu:1 --time 24:00:00 --wrap "train --input-path ./data/filter_dataset_train.h5 --balance-classes --balance-lengths --model-name filter_esm_650m_gap_v2"
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

    sampler = None
    if (args.balance_classes or args.balance_lengths):
        sampler = Sampler(dataset_train, batch_size=args.batch_size, balance_classes=args.balance_classes, balance_lengths=args.balance_lengths, sample_size=20 * len(dataset_train))
    
    model.fit(Datasets(dataset_train, dataset_test), batch_size=args.batch_size, sampler=sampler, epochs=args.epochs, weight_loss=args.weight_loss)
    output_path = os.path.join(args.output_dir, args.model_name + '.pkl')
    model.save(output_path)
    print(f'train: Saved trained model to {output_path}')


def predict():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--model-path', nargs='+', type=str, default=None)
    parser.add_argument('--output-dir', default='./data/predict', type=str)
    parser.add_argument('--models-dir', default='./models', type=str)
    parser.add_argument('--load-labels', action='store_true')
    args = parser.parse_args()

    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path).replace('.h5', '.predict.csv'))   

    for model_path in args.model_path:
        model_name = os.path.basename(model_path).replace('.pkl', '')
        model = Classifier.load(model_path)
        dataset = Dataset.from_hdf(args.input_path, feature_type=model.feature_type, load_labels=args.load_labels)
        model.scale(dataset, fit=False)
        labels, outputs = model.predict(dataset, include_outputs=True)

        df = dict()
        df[f'{model_name}_label'] = labels
        df['id'] = dataset.index 
        for i in range(outputs.shape[-1]): # Iterate over the model predictions for each class, which correspond to a "probability."
            df[f'{model_name}_output_{i}'] = outputs[:, i]
        if args.load_labels: # If the Dataset is labeled, include the labels in the output. 
            df['label'] = dataset.labels.numpy()

        df = pd.DataFrame(df).set_index('id')

        if os.path.exists(output_path):
            df_ = pd.read_csv(output_path, index_col=0)
            df_ = df_.drop(columns=df.columns, errors='ignore')
            assert np.all(df_.index == df.index), f'predict: Failed to add new predictions to existing predictions file at {output_path}, indices do not match.'
            df = df.merge(df_, left_index=True, right_index=True, how='left')
        
        df.to_csv(output_path)
        if args.load_labels: # If the dataset is labeled, compute and report the balanced accuracy. 
            print(f'predict: Balanced accuracy on the input dataset is {model.accuracy(dataset)}')
        print(f'predict: Saved model {model_name} predictions on {args.input_path} to {output_path}')

