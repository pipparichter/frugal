import numpy as np 
import pandas as pd 
from src.dataset import Dataset, Datasets, update_metadata
from src.split import ClusterStratifiedShuffleSplit
from src.classifier import Classifier
from src.graph import RadiusNeighborsGraph
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


def write_predict(df:pd.DataFrame, path:str):
    if os.path.exists(path):
        df_ = pd.read_csv(path, index_col=0) # Drop any overlapping columns. 
        df_ = df_.drop(columns=df.columns, errors='ignore')
        assert np.all(df_.index == df.index), f'write_predict: Failed to add new predictions to existing predictions file at {path}, indices do not match.'
        df = df.merge(df_, left_index=True, right_index=True, how='left')
    df.to_csv(path)
    print(f'write_predict: Added new predictions to file at {path}.')


def cluster():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title='cluster', dest='subcommand', required=True)

    cluster_parser = subparser.add_parser('fit')
    cluster_parser.add_argument('--dataset-path', type=str)
    cluster_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    cluster_parser.add_argument('--n-clusters', default=10000, type=int)
    cluster_parser.add_argument('--bisecting-strategy', default='largest_non_homogenous', type=str)
    cluster_parser.add_argument('--dims', type=int, default=None)

    cluster_parser = subparser.add_parser('predict')
    cluster_parser.add_argument('--input-path', type=str)
    cluster_parser.add_argument('--output-path', default=None, type=str)
    cluster_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    cluster_parser.add_argument('--cluster-path', default=None, type=str)
    cluster_parser.add_argument('--n', default=10, type=int)

    cluster_parser = subparser.add_parser('metric')
    cluster_parser.add_argument('--dataset-path', type=str, default=None)
    cluster_parser.add_argument('--output-path', type=str, default=None)
    cluster_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    cluster_parser.add_argument('--cluster-path', default=None, type=str)
    cluster_parser.add_argument('--silhouette', action='store_true')
    cluster_parser.add_argument('--silhouette-sample-size', default=5000, type=int)
    cluster_parser.add_argument('--dunn', action='store_true')
    cluster_parser.add_argument('--dunn-inter-dist-method', default='pairwise', type=str)
    cluster_parser.add_argument('--dunn-intra-dist-method', default='center', type=str)
    cluster_parser.add_argument('--davies-bouldin', action='store_true')
    cluster_parser.add_argument('--min-inter-dist', action='store_true')
    cluster_parser.add_argument('--min-inter-dist-method', default='center', type=str)
    cluster_parser.add_argument('--intra-dist', action='store_true')
    cluster_parser.add_argument('--intra-dist-method', default='pairwise', type=str)

    args = parser.parse_args()
    
    if args.subcommand == 'fit':
        cluster_fit(args)
    if args.subcommand == 'predict':
        cluster_predict(args)
    if args.subcommand == 'metric':
        cluster_metric(args)


def cluster_metric(args):
    output_path = args.dataset_path.replace('.h5', '_cluster_metadata.csv') if (args.output_path is None) else args.output_path

    dataset = Dataset.from_hdf(args.dataset_path, feature_type=args.feature_type, attrs=['label', 'cluster_id'])
    clusterer = Clusterer.load(args.cluster_path)
    n_clusters = clusterer.n_clusters

    cluster_metadata_df = pd.DataFrame(index=pd.Index(np.arange(n_clusters), name='cluster_id'))
    if os.path.exists(output_path):
        cluster_metadata_df = pd.read_csv(output_path, index_col=0)
        assert len(cluster_metadata_df) == n_clusters, 'cluster_metric: The length of the cluster metadata DataFrame should be equal to the number of clusters.'

    if args.silhouette:
        sample_size = min(len(dataset) - 1, args.sample_size)
        silhouette_index, cluster_metadata_df_ = clusterer.get_silhouette_index(dataset, sample_size=sample_size) 
        print('cluster_metric: Silhouette index is', silhouette_index)

    elif args.dunn:
        dunn_index, cluster_metadata_df_ = clusterer.get_dunn_index(dataset, inter_dist_method=args.inter_dist_method, intra_dist_method=args.intra_dist_method) 
        print('cluster_metric: Dunn index is', dunn_index)

    elif args.davies_bouldin:
        davies_bouldin_index, cluster_metadata_df_ = clusterer.get_davies_bouldin_index(dataset) 
        print('cluster_metric: Davies-Bouldin index is', davies_bouldin_index)

    # For inter-cluster distances, always returns the smallest non-self value computed acoss all clusters. 
    elif args.min_inter_dist:
        method = args.inter_dist_method
        assert method in clusterer.inter_dist_methods, f'cluster_metric: {method} is not a valid method for computing inter-cluster distances.'
        min_inter_cluster_distance, cluster_metadata_df_ = clusterer.get_min_inter_cluster_distance(dataset, method=method) 
        print(f'cluster_metric: Mean minimum inter-cluster distance using method {method} is', min_inter_cluster_distance)

    elif args.intra_dist:
        method = args.intra_dist_method
        assert method in clusterer.intra_dist_methods, f'cluster_metric: {method} is not a valid method for computing intra-cluster distances.'
        intra_cluster_distance, cluster_metadata_df_ = clusterer.get_intra_cluster_distance(dataset, method=method) 
        print(f'cluster_metric: Mean intra-cluster distance using method {method} is', intra_cluster_distance)


    print(f'cluster_metric: Writing cluster metadata to {output_path}')
    for col in cluster_metadata_df_.columns: # Add the new cluster metadata. 
        cluster_metadata_df[col] = cluster_metadata_df_[col]
    cluster_metadata_df.to_csv(output_path)


# sbatch --mem 300GB --time 10:00:00 --mail-user prichter@caltech.edu --mail-type ALL --output dataset_cluster.out --wrap "cluster fit --dataset-path ./data/datasets/dataset.h5"
def cluster_fit(args):
    
    base_cluster_path = args.dataset_path.replace('.h5', '')

    if not os.path.exists(base_cluster_path + '_cluster.csv'):
        dataset = Dataset.from_hdf(args.dataset_path, feature_type=args.feature_type, attrs=['label'])

        clusterer = Clusterer(n_clusters=args.n_clusters, dims=args.dims, bisecting_strategy=args.bisecting_strategy)
        clusterer.fit(dataset)
        print(f'cluster_fit: {len(dataset)} input sequences sorted into {clusterer.n_clusters} clusters.')

        # Too computationally-intensive to compute the entire distance matrix when clustering on a big dataset.
        df = pd.DataFrame(index=pd.Series(dataset.index, name='id'))
        df['cluster_id'] = clusterer.cluster_ids
        
        clusterer.save(base_cluster_path + '_cluster.pkl') # Save the Clusterer object.
        df.to_csv(base_cluster_path + '_cluster.csv') # Write the cluster predictions to a separate file. 
    else: 
        df = pd.read_csv(base_cluster_path + '_cluster.csv', index_col=0)
        print(f'cluster_fit: Loading existing cluster results from {base_cluster_path + '_cluster.csv'}')

    update_metadata(args.dataset_path, cols=[df.cluster_id]) # Add the cluster ID to dataset file metadata. 


def cluster_predict(args):

    output_path = args.input_path.replace('.h5', '_cluster_predict.csv') if (args.output_path is None) else args.output_path
    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, attrs=[])

    clusterer = Clusterer.load(args.cluster_path)
    dists = clusterer.transform(dataset) # This handles the embedding scaling. 

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

    df.to_csv(output_path)


def dataset():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title='dataset', dest='subcommand', required=True)

    dataset_parser = subparser.add_parser('update')
    dataset_parser.add_argument('--dataset-path', type=str)
    dataset_parser.add_argument('--input-path', type=str)
    dataset_parser.add_argument('--columns', type=str, default=None)
    
    dataset_parser = subparser.add_parser('split')
    dataset_parser.add_argument('--dataset-path', type=str)
    dataset_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)

    dataset_parser = subparser.add_parser('graph')
    dataset_parser.add_argument('--input-path', type=str)
    dataset_parser.add_argument('--output-path', default=None)
    dataset_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    dataset_parser.add_argument('--radius', default=15, type=float)
    
    args = parser.parse_args()
    
    if args.subcommand == 'update':
        dataset_update(args)
    if args.subcommand == 'split':
        dataset_split(args)
    if args.subcommand == 'graph':
        dataset_graph(args)


def dataset_graph(args):

    output_path = args.input_path.replace('.h5', '_graph.pkl') if (args.output_path is None) else args.output_path
    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, attrs=None) # Make sure to load all metadata. 
    graph = RadiusNeighborsGraph(radius=args.radius)
    graph.fit(dataset)
    print(f'dataset_graph: Writing radius neighbors graph with radius {args.radius} to {output_path}')
    graph.save(output_path)


def dataset_update(args):

    df = pd.read_csv(args.input_path, index_col=0)
    columns = args.columns.split(',') if (args.columns is not None) else df.columns 
    update_metadata(args.dataset_path, cols=[df[col] for col in columns])


def dataset_split(args):

    output_dir = os.path.dirname(args.dataset_path)
    output_base_path = os.path.join(output_dir, os.path.basename(args.dataset_path).replace('.h5', ''))

    dataset = Dataset.from_hdf(args.dataset_path, feature_type=args.feature_type, attrs=None) # Make sure to load all metadata. 
    splits = ClusterStratifiedShuffleSplit(dataset, n_splits=1, test_size=0.2, train_size=0.8)
    train_dataset, test_dataset = list(splits)[0]

    print(f'dataset_split: Writing split datasets to {output_dir}.')
    train_dataset.to_hdf(output_base_path + '_train.h5')
    test_dataset.to_hdf(output_base_path + '_test.h5')

    print(f'dataset_split: Writing dataset metadata to output directory {output_dir}.')
    train_dataset.metadata().to_csv(output_base_path + '_train.csv')
    test_dataset.metadata().to_csv(output_base_path + '_test.csv')



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
    paths = args.input_path
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



# sbatch --mail-user prichter@caltech.edu --output model_v1.out --mail-type ALL --mem 80GB --partition gpu --gres gpu:1 --time 100:00:00 --wrap "model fit --dims 1280,1024,2 --input-path ./data/dataset_train.h5 --model-name model_v1"
# sbatch --mail-user prichter@caltech.edu --output model_v2.out --mail-type ALL --mem 80GB --partition gpu --gres gpu:1 --time 100:00:00 --wrap "model fit --dims 1280,1024,512,2 --input-path ./data/dataset_train.h5 --model-name model_v2"
# sbatch --mail-user prichter@caltech.edu --output model_v3.out --mail-type ALL --mem 80GB --partition gpu --gres gpu:1 --time 100:00:00 --wrap "model fit --dims 1280,1024,512,256,2 --input-path ./data/dataset_train.h5 --model-name model_v3"
def model_fit(args):

    model_path = os.path.join(args.output_dir, args.model_name + '.pkl')
    dataset = Dataset.from_hdf(args.dataset_path, feature_type=args.feature_type, attrs=['cluster_id', 'label'])

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
    model_parser.add_argument('--dataset-path', type=str)
    model_parser.add_argument('--model-name', type=str)
    model_parser.add_argument('--output-dir', default='./models', type=str)
    model_parser.add_argument('--feature-type', default='esm_650m_gap', type=str)
    model_parser.add_argument('--dims', type=str, default='1280,1024,512,2')
    model_parser.add_argument('--epochs', default=50, type=int)
    model_parser.add_argument('--batch-size', default=16, type=int)
    model_parser.add_argument('--n-splits', default=1, type=int)
    model_parser.add_argument('--include-viruses', action='store_true')


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


