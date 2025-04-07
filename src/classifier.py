from tqdm import tqdm
import sys
import torch
import os
import numpy as np
import pandas as pd
import torch.nn.functional
from torch.utils.data import DataLoader
import sklearn
from sklearn.preprocessing import StandardScaler
import copy
import pickle
from sklearn.metrics import balanced_accuracy_score
import io
import warnings 
import random
from random import choices 

# TODO: Read more about model weight initializations. Maybe I want to use something other than random? 
# TODO: Why bother scaling the loss function weights by the number of classes? I think it's just a minor thing, so that regardless of the number of
#   classes, if the dataset is balanced, the inverse frequencies are 1 regardless of the number of classes .
# TODO: Refresh my memory on cross-entropy loss. 
# TODO: Find out what the epsilon parameter of the optimizer does. Apparently it's just a small constant added for numerical stability, so there's no division by zero errors. 
#   I think to fully understand why it's necessary, I'll need to read the Adam paper https://arxiv.org/abs/1412.6980 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seed everything for reproducibility. This should be run every time the module is imported. 
seed = 42 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Unpickler(pickle.Unpickler): # https://github.com/pytorch/pytorch/issues/16797
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), weights_only=False, map_location='cpu')
        else: return super().find_class(module, name)


class Sampler():

    def __init__(self, dataset, batch_size:int=16, balance:bool=False, sample_size:int=None, **kwargs):

        self.labels = dataset.label
        self.n_total = len(dataset)
        self.n_classes = dataset.n_classes
        self.idxs = np.arange(len(dataset))
        self.balance = balance

        if balance:
            self.class_n_per_batch = batch_size // self.n_classes
            self.batch_size = self.class_n_per_batch * self.n_classes 
            if self.batch_size != batch_size:
                warnings.warn(f'Sampler.__init__: Specified batch size {batch_size} is not divisible by {self.n_classes}. Using batch size {self.batch_size} instead.')
        else:
            self.class_weights = np.ones(len(self.idxs))
            self.batch_size = batch_size

        self.sample_size = len(dataset) * self.n_classes if (sample_size is None) else sample_size
        self.n_batches = self.sample_size // batch_size + 1
        self.batches = self._get_batches()
        
        coverage = ', '.join([f'({i}) {100 * self._coverage(i):.2f}%' for i in range(self.n_classes)])
        print(f'Sampler.__init__: Generated {self.n_batches} batches of size {self.batch_size}. Coverage per class is {coverage}.')

    def _coverage(self, label:int=1):
        '''Get the fraction of the specified class which is covered by the sampler.'''
        idxs = np.unique(np.concatenate(self.batches)) # Get all unique indices covered by the sampler. 
        batch_n = (self.labels[idxs] == label).sum()
        total_n = (self.labels == label).sum()
        return (batch_n / total_n).item()

    def __iter__(self):
        for i in range(self.n_batches):
            yield self.batches[i]

    def _get_batches(self):
        idxs = {i:self.idxs[self.labels == i] for i in range(self.n_classes)} # Pre-sort the indices and weights to avoid repeated filtering (e.g. self.idxs[self.labels == i]).
        batches = [choices(idxs[i], k=self.class_n_per_batch * self.n_batches) for i in range(self.n_classes)]
        batches = np.concatenate([np.split(np.array(idxs_), self.n_batches) for idxs_ in batches], axis=1)
        return batches

    def __len__(self):
        return self.sample_size
    

class WeightedCrossEntropyLoss(torch.nn.Module):

    def __init__(self, n_classes:int=2, weights:list=None):

        super(WeightedCrossEntropyLoss, self).__init__()

        self.user_specified_weights = (weights is not None)
        weights = [1] * n_classes if (weights is None) else weights
        assert len(weights) == n_classes, f'WeightedCrossEntropyLoss.__init__: Specified weights must be equal to the number of classes, {n_classes}.'
        
        self.weights = torch.FloatTensor(weights).to(DEVICE)
        self.to(DEVICE) # Not actually sure if this is necessary. 

    def fit(self, dataset):
        '''Compute the weights to use based on the inverse frequencies of each class. '''
        if self.user_specified_weights: # Make sure the user knows that specified weights in __init__ are being overwritten. 
            warnings.warn('WeightedCrossEntropyLoss.fit: Fitting the loss function is overriding user-specified weights.')

        n_per_class = [(dataset.label == i).sum() for i in range(dataset.n_classes)]
        self.weights = torch.FloatTensor([(len(dataset) / (n_i * dataset.n_classes)) for n_i in n_per_class]).to(DEVICE)

    def forward(self, outputs, targets):
  
        outputs = outputs.view(targets.shape) # Make sure the outputs and targets have the same shape. Use view to avoid copying. 
        loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') # Reduction specifies the pooling to apply to the output. If 'none', no reduction will be applied. 
        weights = torch.unsqueeze(self.weights, 0).repeat(len(outputs), 1).to(DEVICE)
        weights = (targets * weights).sum(axis=1)

        return (loss * weights).mean()


class Classifier(torch.nn.Module):

    copy_attrs = ['loss_func', 'scaler', 'epochs', 'metrics', 'best_epoch', 'batch_size', 'lr', 'best_weights'] # Attributes to port over when copying. 

    def __init__(self, dims:tuple=(1024, 512, 256, 128, 2), loss_func_weights:list=None, feature_type:str=None):

        super(Classifier, self).__init__()
       
        self.dtype = torch.float32
        self.n_classes = dims[-1]
        self.feature_type = feature_type
        self.dims = dims 
        self.loss_func_weights = loss_func_weights

        layers = list()
        dims = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        for input_dim, output_dim in dims:
            layers.append(torch.nn.Linear(input_dim, output_dim, dtype=self.dtype))
            layers.append(torch.nn.ReLU()) # Don't want the last activation function in model, softmax is included in the loss function. 
        self.model = torch.nn.Sequential(*layers[:-1]) # Initialize the sequential model. 

        self.loss_func = WeightedCrossEntropyLoss(n_classes=self.n_classes, weights=loss_func_weights)
        self.scaler = StandardScaler()
        self.to(DEVICE)

        self._init_metrics()

        # To be populated during model fitting. 
        self.best_epoch = None
        self.epochs = None
        self.batch_size = None
        self.lr = None
        self.best_weights = copy.deepcopy(self.state_dict())


    def _init_metrics(self):
        self.metrics = {'train_loss':[], 'test_accuracy':[-np.inf]}
        self.metrics.update({f'test_precision_{i}':[] for i in range(self.n_classes)})
        self.metrics.update({f'test_recall_{i}':[] for i in range(self.n_classes)})

    def copy(self):
        model = Classifier(dims=self.dims, loss_func_weights=self.loss_func_weights, feature_type=self.feature_type)
        for attr in Classifier.copy_attrs:
            setattr(model, attr, copy.deepcopy(getattr(self, attr)))
        model.load_state_dict(copy.deepcopy(self.state_dict()))
        return model

    def _update_metrics(self, dataset, losses:list=None):
        model_labels = self.predict(dataset, include_outputs=False) # Avoid re-computing the model labels for every metric. 

        if losses is not None: # Only try computing the loss if we are past the first epoch. 
            self.metrics['train_loss'] += [np.mean(losses)]

        self.metrics['test_accuracy'] += [self.accuracy(dataset, model_labels=model_labels)]
        for i in range(self.n_classes):
            self.metrics[f'test_precision_{i}'] += [self.precision(dataset, model_labels=model_labels, class_=i)]
            self.metrics[f'test_recall_{i}'] += [self.recall(dataset, model_labels=model_labels, class_=i)]

        metrics = ['test_accuracy', 'test_precision_0', 'test_recall_0'] # Metrics to show in the progress bar.
        metrics = [f'{metric}={self.metrics[metric][-1]:.3f}' for metric in metrics]
        return ', '.join(metrics)

    def forward(self, inputs:torch.FloatTensor):
        return self.model(inputs) 

    def fitted(self):
        return (self.epochs is not None)
    
    def get_best_metric(self, metric:str='test_precision_0'):
        '''Get the value of the specified metric at the best epoch.'''
        metrics = self.metrics[metric]
        assert len(metrics) > 0, f'Classifier.get_best_metric: There are no stored values for metric {metric}.'
        return metrics[self.best_epoch]
    
    def get_latest_metric(self, metric:str='test_precision_0'):
        '''Get the value of the specified metric at the most recent epoch.'''
        metrics = self.metrics[metric]
        assert len(metrics) > 0, f'Classifier.get_best_metric: There are no stored values for metric {metric}.'
        return metrics[-1]

    def predict(self, dataset, include_outputs:bool=False) -> pd.DataFrame:
 
        assert dataset.scaled, 'Classifier.predict: The input Dataset has not been scaled.'

        self.eval() # Put the model in evaluation mode. This changes the forward behavior of the model (e.g. disables dropout).
        with torch.no_grad(): # Turn off gradient computation, which reduces memory usage. 
            outputs = self(dataset.embedding) # Run a forward pass of the model.
            outputs = torch.nn.functional.softmax(outputs, 1) # Apply sigmoid activation, which is applied as a part of the loss function during training. 
            outputs = outputs.cpu().numpy()
        
        labels = np.argmax(outputs, axis=1).ravel() # Convert out of one-hot encodings.
        return (labels, outputs) if include_outputs else labels

    # I care about the ratio of false negatives relative to the total number of negative instances (how frequently is the model marking
    # something as spurious when it is not?). Also, there are a lot of This is FN / (FN + TN), or (1 - recall)
    def accuracy(self, dataset, model_labels:np.ndarray=None) -> float:
        '''Compute the balanced accuracy of the model on the input dataset.'''
        labels = dataset.label # Get the non-one-hot encoded labels from the dataset. 
        model_labels = self.predict(dataset) if (model_labels is None) else model_labels

        return balanced_accuracy_score(labels, model_labels)

    def recall(self, dataset, class_:int=0, model_labels:np.ndarray=None) -> float:
        '''Compute the recall for a particular class on the input dataset, i.e. the ability of the model
        to correctly-identify instances of that class.'''
        labels = dataset.label 
        model_labels = self.predict(dataset) if (model_labels is None) else model_labels

        n = ((model_labels == class_) & (labels == class_)).sum()
        N = (labels == class_).sum() # Total number of relevant instances (i.e. members of the class)
        return n / N if (N > 0) else 0

    def precision(self, dataset, class_:int=0, model_labels:np.ndarray=None):
        '''Compute the precision for a particular class on the input dataset, i.e. the ability of the model
        to correctly distinguish between the classes.'''
        labels = dataset.label 
        model_labels = self.predict(dataset) if (model_labels is None) else model_labels

        n = ((model_labels == class_) & (labels == class_)).sum()
        N = (model_labels == class_).sum() # Total number of retrieved instances (i.e. predicted members of the class)
        return n / N if (N > 0) else 0
    
    def scale(self, dataset, fit:bool=True):
        # Repeatedly scaling a Dataset causes problems, though I am not sure why. I would have thought that
        # subsequent applications of a scaler would have no effect. 
        assert not dataset.scaled, 'Classifier.scale: The dataset has already been scaled.'

        embedding = dataset.embedding.cpu().numpy()
        if fit: # If specified, fit the scaler first. 
            self.scaler.fit(embedding)
        embedding = self.scaler.transform(embedding)
        dataset.embedding = torch.FloatTensor(embedding).to(DEVICE) 
        dataset.scaled = True

    def __gt__(self, model):
        return self.get_best_metric('test_accuracy') > model.get_best_metric('test_accuracy')

    def load_best_weights(self):
        self.load_state_dict(self.best_weights) # Load the best model weights. 

    def fit(self, datasets:tuple, epochs:int=100, lr:float=1e-7, batch_size:int=16, fit_loss_func:bool=False):

        assert datasets.test.scaled, 'Classifier.fit: The input test Dataset has not been scaled.' 
        assert datasets.train.scaled, 'Classifier.fit: The input train Dataset has not been scaled.'

        self.train() # Put the model in train mode.

        if fit_loss_func: 
            self.loss_func.fit(datasets.train) # Set the weights of the loss function.

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        sampler = Sampler(datasets.train, batch_size=batch_size, balance=True, sample_size=10 * len(datasets.train))
        dataloader = DataLoader(datasets.train, batch_sampler=sampler)

        self.best_epoch = 0
        self._init_metrics()
        self._update_metrics(datasets.test) # Initialize the metrics list. 

        for epoch in range(epochs):
            losses = list() # Re-initialize the epoch loss. 
            for batch in dataloader:
                # Evaluate the model on the batch in the training dataloader. 
                outputs, targets = self(batch['embedding']), batch['label_one_hot_encoded'] 
                loss = self.loss_func(outputs, targets)
                loss.backward() 
                losses.append(loss.item()) # Store each loss for computing metrics. 
                
                optimizer.step()
                optimizer.zero_grad()
            
            metrics = self._update_metrics(datasets.test, losses=losses)

            if self.get_latest_metric('test_accuracy') > self.get_best_metric('test_accuracy'):
                self.best_epoch = epoch + 1
                self.best_weights = copy.deepcopy(self.state_dict())
                print(f'Classifier.fit: New best model weights found after epoch {epoch}. {metrics}', flush=True)
            else:
                print(f'Classifier.fit: No improvement after epoch {epoch}. {metrics}', flush=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    @classmethod
    def load(cls, path:str):
        with open(path, 'rb') as f:
            # obj = pickle.load(f)
            obj = Unpickler(f).load()
        return obj.to(DEVICE)  

    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    # def get_dims(self) -> list:
    #     '''Get the input and output dimensions of each linear layer.'''
    #     dims = list()
    #     for param in self.parameters():
    #         shape = param.shape
    #         if len(shape) == 2: # Activation parameters only have one dimension. 
    #             output_dim, input_dim = shape 
    #             dims.append((input_dim, output_dim))
    #     return dims 
