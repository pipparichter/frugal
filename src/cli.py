import numpy as np 
import pandas as pd 
from src.dataset import Dataset, Datasets, Pruner, update_metadata
from src.split import ClusterStratifiedShuffleSplit
from src.classifier import Classifier
import argparse
from src.clusterer import Clusterer, get_cluster_metadata
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


def write_predict(df:pd.DataFrame, path:str):
    if os.path.exists(path):
        df_ = pd.read_csv(path, index_col=0) # Drop any overlapping columns. 
        df_ = df_.drop(columns=df.columns, errors='ignore')
        assert np.all(df_.index == df.index), f'write_predict: Failed to add new predictions to existing predictions file at {path}, indices do not match.'
        df = df.merge(df_, left_index=True, right_index=True, how='left')
    df.to_csv(path)
    print(f'write_predict: Added new predictions to file at {path}.')


def cluster_metadata(args):

    output_path = args.input_path.replace('.h5', 'cluster_metadata.csv') if (args.output_path is None) else args.output_path
    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, attrs=['label', 'cluster_id'])
    clusterer = Clusterer.load(args.cluster_path)
    cluster_metadata_df = get_cluster_metadata(dataset, clusterer)
    print(f'cluster_metadata: Cluster metadata written to {output_path}')


def cluster_fit(args):
    
    base_output_path = args.input_path.replace('.h5', '')

    if not os.path.exists(base_output_path + '_cluster.csv'):
        dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, attrs=['label'])

        clusterer = Clusterer(n_clusters=args.n_clusters, verbose=args.verbose, bisecting_strategy=args.bisecting_strategy)
        clusterer.fit(dataset)
        print(f'cluster_fit: {len(dataset)} input sequences sorted into {clusterer.n_clusters} clusters.')

        # Too computationally-intensive to compute the entire distance matrix when clustering on a big dataset.
        df = pd.DataFrame(index=pd.Series(dataset.index, name='id'))
        df['cluster_id'] = clusterer.cluster_ids
        
        clusterer.save(base_output_path + '_cluster.pkl') # Save the Clusterer object.
        df.to_csv(base_output_path + '_cluster.csv') # Write the cluster predictions to a separate file. 
    else: 
        df = pd.read_csv(base_output_path + '_cluster.csv', index_col=0)
        print(f'cluster_fit: Loading existing cluster results from {base_output_path + '_cluster.csv'}')

    update_metadata(args.input_path, df.cluster_id) # Add the cluster ID to dataset file metadata. 


     
def cluster_predict(args):

    output_path = args.input_path.replace('.h5', '_cluster_predict.csv') if (args.output_path is None) else args.output_path
    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, attrs=[])

    clusterer = Clusterer.load(args.cluster_path)
    dists = clusterer.transform(dataset)

    top_n_cluster_ids = np.argsort(dists, axis=1)[:, :args.n]
    top_n_dists = np.array([dists[i, top_n_cluster_ids[i]] for i in range(top_n_cluster_ids.shape[0])])

    df = pd.DataFrame(index=pd.Index(dataset.index, name='id'))
    df['top_cluster_id'] = top_n_cluster_ids[:, 0]
    df['top_cluster_distance'] = top_n_dists[:, 0]
    df['top_cluster_label'] = df['top_cluster_id'].map(clusterer.get_cluster_id_to_label_map())
    for i in range(2, args.n):
        df[f'rank_{i}_cluster_id'] = top_n_cluster_ids[:, i]
        df[f'rank_{i}_cluster_distance'] = top_n_dists[:, i] 
        df[f'rank_{i}_cluster_label'] = df[f'rank_{i}_cluster_id'].map(clusterer.get_cluster_id_to_label_map())

    write_predict(df, output_path)


def cluster():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title='cluster', dest='subcommand', required=True)

    cluster_parser = subparser.add_parser('fit')
    cluster_parser.add_argument('--input-path', type=str)
    cluster_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    cluster_parser.add_argument('--n-clusters', default=50000, type=int)
    cluster_parser.add_argument('--bisecting-strategy', default='largest_non_homogenous', type=str)
    cluster_parser.add_argument('--verbose', action='store_true')

    cluster_parser = subparser.add_parser('predict')
    cluster_parser.add_argument('--input-path', type=str)
    cluster_parser.add_argument('--output-path', default=None, type=str)
    cluster_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    cluster_parser.add_argument('--cluster-path', default=None, type=str)
    cluster_parser.add_argument('--n', default=10, type=int)

    cluster_parser = subparser.add_parser('metadata')
    cluster_parser.add_argument('--input-path', type=str, default=None)
    cluster_parser.add_argument('--output-path', default=None, type=str)
    cluster_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    cluster_parser.add_argument('--cluster-path', default=None, type=str)

    args = parser.parse_args()
    
    if args.subcommand == 'fit':
        cluster_fit(args)
    if args.subcommand == 'predict':
        cluster_predict(args)
    if args.subcommand == 'metadata':
        cluster_metadata(args)


def dataset():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title='dataset', dest='subcommand', required=True)

    dataset_parser = subparser.add_parser('update')
    dataset_parser.add_argument('--dataset-path', type=str)
    dataset_parser.add_argument('--input-path', type=str)
    dataset_parser.add_argument('--columns', type=str, default=None)
    
    dataset_parser = subparser.add_parser('split')
    dataset_parser.add_argument('--input-path', type=str)
    dataset_parser.add_argument('--output-dir', default=None)
    dataset_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    
    args = parser.parse_args()
    
    if args.subcommand == 'update':
        dataset_update(args)
    if args.subcommand == 'split':
        dataset_split(args)


def dataset_update(args):

    df = pd.read_csv(args.input_path, index_col=0)
    columns = args.columns.split(',') if (args.columns is not None) else df.columns 
    for col in columns:
        update_metadata(args.dataset_path, df[col])


def dataset_split(args):

    output_dir = os.path.dirname(args.input_path) if (args.output_dir is None) else args.output_dir
    output_base_path = os.path.join(output_dir, os.path.basename(args.input_path).replace('.h5', ''))

    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, attrs=None) # Make sure to load all metadata. 
    splits = ClusterStratifiedShuffleSplit(dataset, n_splits=1, test_size=0.2, train_size=0.8)
    train_dataset, test_dataset = list(splits)[0]

    print(f'train_test_split: Writing split datasets to {output_dir}.')
    train_dataset.to_hdf(output_base_path + '_train.h5')
    test_dataset.to_hdf(output_base_path + '_test.h5')

    print(f'train_test_split: Writing dataset metadata to output directory {output_dir}.')
    train_dataset.metadata().to_csv(output_base_path + '_train.csv')
    test_dataset.metadata().to_csv(output_base_path + '_test.csv')


# def prune():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input-path', type=str)
#     parser.add_argument('--output-path', default=None, type=str)
#     parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
#     parser.add_argument('--overwrite', action='store_true')
#     parser.add_argument('--radius', default=2, type=float)
#     args = parser.parse_args()

#     output_path = args.input_path.replace('.h5', '_dereplicated.h5') if (args.output_path is None) else args.output_path

#     dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, attrs=None) # Load all attributes into the Dataset. 
#     pruner = Pruner(radius=args.radius)
#     pruner.fit(dataset)
#     dataset = pruner.prune(dataset)
#     print(f'prune: Writing dereplicated Dataset to {output_path}')
#     dataset.to_hdf(output_path)
#     dataset.metadata().to_csv(output_path.replace('.h5', '.csv'))

# sbatch --mem 200GB --time 10:00:00 --gres gpu:1 --partition gpu --wrap "library add --input-path GCF_000005845.2_protein.faa GCF_000009045.1_protein.faa GCF_000006765.1_protein.faa GCF_000195955.2_protein.faa --library-dir ../embeddings/"
def library():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title='library', dest='subcommand', required=True)

    library_parser = subparser.add_parser('add')
    library_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    library_parser.add_argument('--max-length', default=2000, type=int)
    library_parser.add_argument('--input-path', nargs='+', default=None)
    library_parser.add_argument('--library-dir', type=str, default='./data/embeddings')
    library_parser.add_argument('--parallelize', action='store_true')
    library_parser.add_argument('--n-processes', default=4, type=int)
    
    library_parser = subparser.add_parser('get')
    library_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    library_parser.add_argument('--input-path', default=None)
    library_parser.add_argument('--library-dir', type=str, default='./data/embeddings')
    library_parser.add_argument('--entry-name-col', type=str, default='genome_id')
    
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

    output_path = args.input_path.replace('.csv', '.h5')
    store = pd.HDFStore(output_path, mode='a')

    lib = EmbeddingLibrary(dir_=args.library_dir, feature_type=args.feature_type) # , max_length=args.max_length)
    
    dtypes = {f'query_{field}':dtype for field, dtype in GBFFFile.dtypes.items()}
    dtypes.update({f'top_hit_{field}':dtype for field, dtype in GBFFFile.dtypes.items()})
    metadata_df = pd.read_csv(args.input_path, index_col=0, dtype=dtypes) # Expect the index column to be the sequence ID. 
    store.put('metadata', metadata_df, format='table')

    embeddings_df = list()
    for entry_name, df in metadata_df.groupby(args.entry_name_col): # Read in the embeddings from the genome file. 
        print(f'library_get: Loading {len(df)} embeddings from {entry_name}_embedding.csv.')
        embeddings_df.append(lib.get(entry_name, ids=df.index))
    embeddings_df = pd.concat(embeddings_df)
    embeddings_df = embeddings_df.loc[metadata_df.index, :] # Make sure the embeddings are in the same order as the metadata. 
    store.put(args.feature_type, embeddings_df, format='table')
    store.close()
    print(f'library_get: Embeddings of type {args.feature_type} written to {output_path}')


# v1 1280,1024,2
# v2 1280,1024,512,2
# v3 1280,1024,512,256,2

# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 300GB --partition gpu --gres gpu:1 --time 100:00:00 --wrap "model fit --dims 1280,1024,2 --input-path ./data/dataset_train.h5 --model-name campylobacterota_v1"
# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 300GB --partition gpu --gres gpu:1 --time 100:00:00 --wrap "model fit --dims 1280,1024,512,2 --input-path ./data/dataset_train.h5 --model-name campylobacterota_v2"
# sbatch --mail-user prichter@caltech.edu --mail-type ALL --mem 300GB --partition gpu --gres gpu:1 --time 100:00:00 --wrap "model fit --dims 1280,1024,512,256,2 --input-path ./data/dataset_train.h5 --model-name campylobacterota_v3"
def model_fit(args):

    model_path = os.path.join(args.output_dir, args.model_name + '.pkl')
    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, attrs=['cluster_id', 'label', 'domain'])

    if not args.include_viruses:
        subset_idxs = np.where(dataset.domain != 'Viruses')[0]
        print(f'model_fit: Excluding {len(dataset) - len(subset_idxs)} viral proteins from the training data.')
        dataset = dataset.subset(subset_idxs)

    dims = [int(d) for d in args.dims.split(',')] if (args.dims is not None) else [dataset.n_features, 512, dataset.n_classes]
    assert dims[0] == dataset.n_features, f'model_fit: First model dimension {dims[0]} does not match the number of features {dataset.n_features}.'
    assert dims[-1] == dataset.n_classes, f'model_fit: Last model dimension {dims[-1]} does not match the number of classes {dataset.n_classes}.'

    splits = ClusterStratifiedShuffleSplit(dataset, n_splits=args.n_splits)
    best_model = None
    for i, (train_dataset, test_dataset) in enumerate(splits):
        model = Classifier(dims=dims, feature_type=args.feature_type)
        model.scale(train_dataset, fit=True)
        model.scale(test_dataset, fit=False)
        model.fit(Datasets(train_dataset, test_dataset), batch_size=args.batch_size, epochs=args.epochs)

        if (best_model is None) or (model > best_model):
            best_model = model.copy()
            best_model.save(model_path)
            print(f'model_fit: New best model found. Saved to {model_path}.')
        print()

    best_model.save(model_path)
    print(f'model_fit: Saved best model to {model_path}')


# def model_tune(args):

#     base_model_path = os.path.join(args.output_dir, args.base_model_name + '.pkl')
#     model_path = os.path.join(args.output_dir, args.model_name + '.pkl')

#     # When fine-tuning, should I just use the same dataset for training and validation? Or just not use a validation set. 
#     model = Classifier.load(base_model_path)
#     dataset = Dataset.from_hdf(args.input_path, feature_type=model.feature_type, attrs=['label'])
#     model.scale(dataset, fit=False) # Use the existing StandardScaler without re-fitting.
#     model.fit(Datasets(dataset, dataset), fit_loss_func=False, batch_size=args.batch_size, epochs=args.epochs)
#     # Don't load the best model weights here. 

#     model.save(model_path)
#     print(f'model_fit: Saved best model to {model_path}')


def model_predict(args):

    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path).replace('.h5', '_predict.csv'))   

    for model_path in args.model_path:
        model_name = os.path.basename(model_path).replace('.pkl', '')
        model = Classifier.load(model_path)
        model.load_best_weights()

        attrs = ['label'] if args.load_labels else []
        dataset = Dataset.from_hdf(args.input_path, feature_type=model.feature_type, attrs=attrs)

        model.scale(dataset, fit=False)
        model_labels, outputs = model.predict(dataset, include_outputs=True)

        df = dict()
        df[f'{model_name}_label'] = model_labels
        df['id'] = dataset.index 
        for i in range(outputs.shape[-1]): # Iterate over the model predictions for each class, which correspond to a "probability."
            df[f'{model_name}_output_{i}'] = outputs[:, i]
        for attr in dataset.attrs: # Add all dataset attributes to the DataFrame. 
            df[attr] = getattr(dataset, attr)
        df = pd.DataFrame(df).set_index('id')

        write_predict(df, output_path)
        

def model():

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title='model', dest='subcommand', required=True)

    model_parser = subparser.add_parser('fit')
    model_parser.add_argument('--input-path', type=str)
    model_parser.add_argument('--cluster-path', type=str, default='./data/dataset_dereplicated_cluster.csv')
    model_parser.add_argument('--model-name', type=str)
    model_parser.add_argument('--output-dir', default='./models', type=str)
    model_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    model_parser.add_argument('--dims', type=str, default='1280,1024,512,2')
    model_parser.add_argument('--epochs', default=100, type=int)
    model_parser.add_argument('--batch-size', default=16, type=int)
    model_parser.add_argument('--n-splits', default=1, type=int)
    model_parser.add_argument('--include-viruses', action='store_true')


    # model_parser = subparser.add_parser('tune')
    # model_parser.add_argument('--input-path', type=str)
    # model_parser.add_argument('--base-model-name', default=None, type=str)
    # model_parser.add_argument('--model-name', default=None, type=str)
    # model_parser.add_argument('--output-dir', default='./models', type=str)
    # model_parser.add_argument('--epochs', default=50, type=int)
    # model_parser.add_argument('--batch-size', default=16, type=int)
    # model_parser.add_argument('--n-splits', default=5, type=int)


    model_parser = subparser.add_parser('predict')
    model_parser.add_argument('--input-path', type=str)
    model_parser.add_argument('--model-path', nargs='+', type=str, default=None)
    model_parser.add_argument('--output-dir', default='./data/results/', type=str)
    model_parser.add_argument('--load-labels', action='store_true')

    args = parser.parse_args()
    
    if args.subcommand == 'fit':
        model_fit(args)
    if args.subcommand == 'predict':
        model_predict(args)
    # if args.subcommand == 'tune':
    #     model_tune(args)

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



# def stats():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset-path', type=str, default=None)
#     parser.add_argument('--model-path', type=str, default=None)
#     args = parser.parse_args()

#     if args.model_path is not None:
#         model_name = os.path.basename(args.model_path).replace('.pkl', '')
#         model = Classifier.load(args.model_path)

#         print('stats:', model_name)
#         print('stats:')
#         print('stats: Model dimensions', ' > '.join([str(dims) for dims in model.get_dims()]))
#         print(f'stats: Trained for {model.epochs} epochs with batch size {model.batch_size} and learning rate {model.lr}.')
#         print(f'stats: Used cross-entropy loss with weights', [w.item() for w in model.loss_func.weights])
#         print(f'stats: Using weights from epoch {model.best_epoch}, selected using metric {model.metric}.')
#         if model.sampler is not None:
#             print(f'stats: Balanced classes', model.sampler.balance_classes)
#             print(f'stats: Balanced lengths', model.sampler.balance_lengths)
#         metrics = ['test_precision_0', 'test_recall_0', 'test_precision_1', 'test_recall_1', 'test_accuracy']
#         print(f'stats: Metrics')
#         for metric in metrics:
#             print(f'stats:\t{metric} = {model.metrics[metric][model.best_epoch]:.3f}')


