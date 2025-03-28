from src import seed 
seed(42) # Make sure everything is random-seeded for reproducibility. 

import numpy as np 
import pandas as pd 
from src.dataset import Dataset, Datasets, Pruner
from src.split import ClusterStratifiedShuffleSplit
from src.classifier import Classifier
import argparse
from src.clusterer import Clusterer
from src.reference import Reference, ReferenceAnnotator
from src.files import FASTAFile, GBFFFile
from src.embed import get_embedder, EmbeddingLibrary
from src.embed.library import add 
import random
import glob
from tqdm import tqdm 
import os
from src import get_genome_id, fillna
from multiprocessing import Pool
from transformers import logging


logging.set_verbosity_error() # Turn off the warning about uninitialized weights. 

def cluster():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', default=None, type=str)
    parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--n-clusters', default=None, type=int)
    parser.add_argument('--check-homogenous', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    output_path = args.input_path.replace('.h5', '_cluster.csv') if (args.output_path is None) else args.output_path

    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, attrs=['label'])
    clusterer = Clusterer(n_clusters=args.n_clusters, verbose=args.verbose)
    clusterer.fit(dataset, check_homogenous=args.check_homogenous)
    clusterer.write(output_path, dataset=dataset)

    print(f'cluster: {len(dataset)} input sequences sorted into {clusterer.n_clusters} clusters.')
    # print(f'cluster: {clusterer.n_singleton_clusters} clusters only contain one sequence.')
    print(f'cluster: Sequence clusters saved to {output_path}')


def prune():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', default=None, type=str)
    parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--radius', default=2, type=float)
    args = parser.parse_args()

    output_path = args.input_path.replace('.h5', '_dereplicated.h5') if (args.output_path is None) else args.output_path

    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, attrs=None) # Load all attributes into the Dataset. 
    pruner = Pruner(radius=args.radius)
    pruner.fit(dataset)
    dataset = pruner.prune(dataset)
    print(f'prune: Writing dereplicated Dataset to {output_path}')
    dataset.to_hdf(output_path)
    dataset.to_csv(output_path.replace('.csv', '.h5'), metadata=True)


def library():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title='library', dest='subcommand', required=True)

    parser_library = subparser.add_parser('add')
    parser_library.add_argument('--feature-type', default='esm_650m_gap', type=str)
    parser_library.add_argument('--max-length', default=2000, type=int)
    parser_library.add_argument('--input-path', nargs='+', default=None)
    parser_library.add_argument('--input-dir', type=str, default='./data/proteins/')
    parser_library.add_argument('--library-dir', type=str, default='./data/embeddings')
    parser_library.add_argument('--parallelize', action='store_true')
    parser_library.add_argument('--n-processes', default=4, type=int)
    
    parser_library = subparser.add_parser('get')
    parser_library.add_argument('--feature-type', default='esm_650m_gap', type=str)
    parser_library.add_argument('--input-path', default=None)
    parser_library.add_argument('--output-path', default=None)
    parser_library.add_argument('--library-dir', type=str, default='./data/embeddings')
    parser_library.add_argument('--entry-name-col', type=str, default='genome_id')
    
    args = parser.parse_args()
    
    if args.subcommand == 'add':
        library_add(args)
    if args.subcommand == 'get':
        library_get(args)


# sbatch --mail-user prichter@caltech.edu --mail-type ALL --partition gpu --gpus-per-task 4 --mem-per-gpu 300GB --gres gpu --time 24:00:00 --wrap "library add"
def library_add(args):

    lib = EmbeddingLibrary(dir_=args.library_dir, feature_type=args.feature_type, max_length=args.max_length)
    paths = args.input_path if (args.input_path is not None) else glob.glob(os.path.join(args.input_dir, '*'))
    random.shuffle(paths) # Shuffle so starting multiple processes works better. 

    if args.parallelize:
        # Add a library to the start of each set of arguments. If I didn't copy it, I was getting a "too many open files" error, though I am not sure why.
        inputs = [[lib.copy()] + list(paths_) for paths_ in np.array_split(paths, len(paths) // args.n_processes)]
        pool = Pool(args.n_processes)
        pool.starmap(add, inputs)
        pool.close()
    else:
        add(lib, *paths)


def library_get(args):

    output_path = args.input_path.replace('csv', 'h5') if (args.output_path is None) else args.output_path
    store = pd.HDFStore(output_path, mode='a')

    lib = EmbeddingLibrary(dir_=args.library_dir, feature_type=args.feature_type) # , max_length=args.max_length)
    
    dtypes = {f'query_{field}':dtype for field, dtype in GBFFFile.dtypes.items()}
    dtypes.update({f'top_hit_{field}':dtype for field, dtype in GBFFFile.dtypes.items()})
    df = pd.read_csv(args.input_path, index_col=0, dtype=dtypes) # Expect the index column to be the sequence ID. 
    store.put('metadata', df, format='table')

    embeddings_df = list()
    for entry_name, df_ in df.groupby(args.entry_name_col): # Read in the embeddings from the genome file. 
        print(f'library_get: Loading {len(df_)} embeddings from {entry_name}_embedding.csv.')
        embeddings_df.append(lib.get(entry_name, ids=df_.index))
    embeddings_df = pd.concat(embeddings_df)
    embeddings_df = embeddings_df.loc[df.index, :] # Make sure the embeddings are in the same order as the metadata. 
    store.put(args.feature_type, embeddings_df, format='table')
    store.close()
    print(f'library_get: Embeddings of type {args.feature_type} written to {output_path}')


def ref():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', nargs='+', type=str)
    parser.add_argument('--output-dir', default='./data/ref/', type=str)
    parser.add_argument('--gbffs-dir', default='./data/ncbi/gbffs', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--annotate', action='store_true')
    parser.add_argument('--min-sequence-identity', type=float, default=1)
    parser.add_argument('--max-overlap', type=int, default=50)
    args = parser.parse_args()

    input_paths = args.input_path  
    genome_ids = [get_genome_id(path, errors='raise') for path in input_paths]
    gbff_paths = [os.path.join(args.gbffs_dir, f'{genome_id}_genomic.gbff') for genome_id in genome_ids]
    
    for i, (genome_id, input_path, gbff_path) in enumerate(zip(genome_ids, input_paths, gbff_paths)):
        ref_all_output_path = os.path.join(args.output_dir, f'{genome_id}_ref_all.csv')
        ref_output_path = os.path.join(args.output_dir, f'{genome_id}_ref.csv')

        if (not os.path.exists(ref_output_path)) or args.overwrite:
            print(f'ref: Searching reference for genome {genome_id}, {i} of {len(genome_ids)}.')
            reference = Reference(gbff_path)
            query_df = FASTAFile(path=input_path).to_df(prodigal_output=True)
            ref_all_df, ref_df = reference.search(query_df, verbose=False)
            ref_all_df.to_csv(ref_all_output_path)
            ref_df.to_csv(ref_output_path)
        if args.annotate:
            print(f'ref: Annotating reference results for genome {genome_id}, {i} of {len(genome_ids)}.')
            annotator = ReferenceAnnotator(max_overlap=args.max_overlap, min_sequence_identity=args.min_sequence_identity)
            annotator.run(ref_output_path)
        print()

    print(f'ref: Search complete. Results written to {args.output_dir}')


# v1 1280,1024,2
# v2 1280,1024,512,2
# v3 1280,1024,512,256,2

# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 300GB --partition gpu --gres gpu:1 --time 100:00:00 --wrap "train --dims 1280,1024,2 --input-path ./data/train_dataset.h5 --model-name campylobacterota_v1"
# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 300GB --partition gpu --gres gpu:1 --time 100:00:00 --wrap "train --dims 1280,1024,512,2 --input-path ./data/train_dataset.h5 --model-name campylobacterota_v2"
# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 300GB --partition gpu --gres gpu:1 --time 100:00:00 --wrap "train --dims 1280,1024,512,256,2 --input-path ./data/train_dataset.h5 --model-name campylobacterota_v3"
def train():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--cluster-path', type=str, default=None)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--output-dir', default='./models', type=str)
    parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    parser.add_argument('--dims', type=str, default='1280,1024,512,2')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--n-splits', default=5, type=int)

    args = parser.parse_args()
    output_path = os.path.join(args.output_dir, args.model_name + '.pkl')
    cluster_path = args.input_path.replace('.h5', '_cluster.csv') if (args.cluster_path is None) else args.cluster_path # Define a default cluster path. 

    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, attrs=['label', 'genome_id'])

    dims = [int(d) for d in args.dims.split(',')] if (args.dims is not None) else [dataset.n_features, 512, dataset.n_classes]
    assert dims[0] == dataset.n_features, f'train: First model dimension {dims[0]} does not match the number of features {dataset.n_features}.'
    assert dims[-1] == dataset.n_classes, f'train: Last model dimension {dims[-1]} does not match the number of classes {dataset.n_classes}.'

    splits = ClusterStratifiedShuffleSplit(dataset, cluster_path=cluster_path, n_splits=args.n_splits)
    best_model = None
    best_split = None
    for i, (train_dataset, test_dataset) in enumerate(splits):
        model = Classifier(dims=dims, feature_type=args.feature_type)
        model.scale(train_dataset, fit=True)
        model.scale(test_dataset, fit=False)

        model.fit(Datasets(train_dataset, test_dataset), batch_size=args.batch_size, epochs=args.epochs)

        if (best_model is None) or (model > best_model):
            best_model = model.copy()
            best_split = i
            splits.save(os.path.join(args.output_dir, args.model_name + '_splits.json'), best_split=best_split)
            best_model.save(output_path)
            print(f'train: New best model found. Saved to {output_path}.')
        print()

    best_model.save(output_path)
    splits.save(os.path.join(args.output_dir, args.model_name + '_splits.json'), best_split=best_split)

    print(f'train: Saved best model to {output_path}')


def predict():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--model-path', nargs='+', type=str, default=None)
    parser.add_argument('--output-dir', default='./data/', type=str)
    # parser.add_argument('--models-dir', default='./models', type=str)
    parser.add_argument('--load-labels', action='store_true')
    args = parser.parse_args()

    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path).replace('.h5', '_predict.csv'))   

    for model_path in args.model_path:
        model_name = os.path.basename(model_path).replace('.pkl', '')
        model = Classifier.load(model_path)

        attrs = ['label'] if args.load_labels else []
        dataset = Dataset.from_hdf(args.input_path, feature_type=model.feature_type, attrs=attrs)

        model.scale(dataset, fit=False)
        model_labels, outputs = model.predict(dataset, include_outputs=True)

        df = dict()
        df['model_label'] = model_labels
        df['id'] = dataset.index 
        for i in range(outputs.shape[-1]): # Iterate over the model predictions for each class, which correspond to a "probability."
            df[f'model_output_{i}'] = outputs[:, i]
        for attr in dataset.attrs: # Add all dataset attributes to the DataFrame. 
            df[attr] = getattr(dataset, attr)
        df = pd.DataFrame(df).set_index('id')

        if args.load_labels: # If the dataset is labeled, compute and report the balanced accuracy. 
            conditions = [(df.model_label == 1) & (df.label == 0), (df.model_label  == 1) & (df.label == 1), (df.model_label == 0) & (df.label == 1), (df.model_label  == 0) & (df.label == 0)]
            choices = ['fp', 'tp', 'fn', 'tn']
            df['model_confusion_matrix'] = np.select(conditions, choices, default='none')
            fp, tp, fn, tn = [condition.sum() for condition in conditions]
            print(f'predict: Balanced accuracy {0.5 * (tp / (tp + fn) + (tn / (tn + fp))):.3f}')
            print(f'predict: Recall {tn / (fp + tn):.3f}, {tp / (tp + fn):.3f}')
            print(f'predict: Precision {tn / (fn + tn):.3f}, {tp / (tp + fp):.3f}')

        # Rename the generic model columns to the actual model name. 
        df = df.rename(columns={col:col.replace('model', model_name) for col in df.columns})

        if os.path.exists(output_path):
            df_ = pd.read_csv(output_path, index_col=0) # Drop any overlapping columns. 
            df_ = df_.drop(columns=df.columns, errors='ignore')
            assert np.all(df_.index == df.index), f'predict: Failed to add new predictions to existing predictions file at {output_path}, indices do not match.'
            df = df.merge(df_, left_index=True, right_index=True, how='left')
        
        df.to_csv(output_path)
        print(f'predict: Saved model {model_name} predictions on {args.input_path} to {output_path}')


def stats():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()

    if args.model_path is not None:
        model_name = os.path.basename(args.model_path).replace('.pkl', '')
        model = Classifier.load(args.model_path)

        print('stats:', model_name)
        print('stats:')
        print('stats: Model dimensions', ' > '.join([str(dims) for dims in model.get_dims()]))
        print(f'stats: Trained for {model.epochs} epochs with batch size {model.batch_size} and learning rate {model.lr}.')
        print(f'stats: Used cross-entropy loss with weights', [w.item() for w in model.loss_func.weights])
        print(f'stats: Using weights from epoch {model.best_epoch}, selected using metric {model.metric}.')
        if model.sampler is not None:
            print(f'stats: Balanced classes', model.sampler.balance_classes)
            print(f'stats: Balanced lengths', model.sampler.balance_lengths)
        metrics = ['test_precision_0', 'test_recall_0', 'test_precision_1', 'test_recall_1', 'test_accuracy']
        print(f'stats: Metrics')
        for metric in metrics:
            print(f'stats:\t{metric} = {model.metrics[metric][model.best_epoch]:.3f}')



# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 100GB --partition gpu --gres gpu:1 --time 24:00:00 --wrap "embed --input-path ./data/campylobacterota_dataset_boundary_errors.csv"
def embed():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', default=None, type=str)
    parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    args = parser.parse_args()

    output_path = args.output_path if (args.output_path is not None) else args.input_path.replace('.csv', '.h5')

    # Sort the sequences in order of length, so that the longest sequences are first. This is so that the 
    # embedder works more efficiently, and if it fails, it fails early in the embedding process.
    df = pd.read_csv(args.input_path, index_col=0)
    df = df.iloc[np.argsort(df.seq.apply(len))[::-1]]

    store = pd.HDFStore(output_path, mode='a', table=True) # Should confirm that the file already exists. 
    existing_keys = [key.replace('/', '') for key in store.keys()]

    if 'metadata' in existing_keys:
        df_ = store.get('metadata') # Load existing metadata. 
        assert np.all(df.index == df_.index), 'embed: The input metadata and existing metadata do not match.'
    else: # Storing in table format means I can add to the file later on. 
        store.put('metadata', df, format='table', data_columns=None)

    print(f'embed: Generating embeddings for {args.feature_type}.')
    embedder = get_embedder(args.feature_type)
    embeddings = embedder(df.seq.values.tolist())
    store.put(args.feature_type, pd.DataFrame(embeddings, index=df.index), format='table', data_columns=None) 
    store.close()
    print(f'embed: Embeddings saved to {output_path}.')



        
