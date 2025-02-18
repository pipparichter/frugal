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

# TODO: Read more about model weight initializations. Maybe I want to use something other than random? 
# TODO: Why bother scaling the loss function weights by the number of classes?
# TODO: Refresh my memory on cross-entropy loss. 
# TODO: Find out what the epsilon parameter of the optimizer does. 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Unpickler(pickle.Unpickler):
    # https://github.com/pytorch/pytorch/issues/16797
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), weights_only=False, map_location='cpu')
        else: return super().find_class(module, name)


class WeightedCrossEntropyLoss(torch.nn.Module):

    def __init__(self, n_classes:int=2):

        super(WeightedCrossEntropyLoss, self).__init__()

        self.weights = torch.FloatTensor([1] * n_classes).to(DEVICE)
        self.to(DEVICE) # Not actually sure if this is necessary. 

    def fit(self, dataset):
        '''Compute the weights to use based on the inverse frequencies of each class. '''
        n_per_class = [(dataset.labels == i).sum() for i in range(dataset.n_classes)]
        self.weights = torch.FloatTensor([(len(dataset) / (n_i * dataset.n_classes)) for n_i in n]).to(DEVICE)

    def forward(self, outputs, targets):
        '''Compute the weighted loss between the targets and outputs. 

        :param outputs: A Tensor of size (batch_size, n_classes). All values should be between 0 and 1, and sum to 1. 
        :param targets: A Tensor of size (batch_size, n_classes), which is a one-hot encoded vector of the labels.  
        '''
        outputs = outputs.view(targets.shape) # Make sure the outputs and targets have the same shape. Use view to avoid copying. 
        loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') # Reduction specifies the pooling to apply to the output. If 'none', no reduction will be applied. 

        weights = torch.unsqueeze(self.weights, 0).repeat(len(outputs), 1)
        weights = (targets * weights).sum(axis=1)

        return (loss * weights).mean()


class Classifier(torch.nn.Module):

    def __init__(self, dims:tuple=(1024, 512, 2)):

        super(Classifier, self).__init__()
       
        self.dtype = torch.float32

        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1], dtype=self.dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[1], dims[2], dtype=self.dtype))

        self.loss_func = WeightedCrossEntropyLoss(n_classes=dims[2])
        self.scaler = StandardScaler()
        self.to(DEVICE)

        self.metrics = dict()
        self.metrics['train_loss'] = []
        self.metrics['test_acc'] = []

    def forward(self, inputs:torch.FloatTensor):
        return self.model(inputs) 

    def predict(self, dataset, include_outputs:bool=False) -> pd.DataFrame:
 
        assert dataset.scaled, 'Classifier.predict: The input Dataset has not been scaled.'

        self.eval() # Put the model in evaluation mode. This changes the forward behavior of the model (e.g. disables dropout).
        dataset = dataset.scale(self.scaler)
        with torch.no_grad(): # Turn off gradient computation, which reduces memory usage. 
            outputs = self(dataset.embeddings) # Run a forward pass of the model.
            outputs = torch.nn.functional.softmax(outputs, 1) # Apply sigmoid activation, which is applied as a part of the loss function during training. 
            outputs = outputs.cpu().numpy()
        
        labels = np.argmax(outputs, axis=1).ravel() # Convert out of one-hot encodings.
        return (labels, outputs) if include_outputs else labels


    def accuracy(self, dataset) -> float:
        '''Compute the balanced accuracy of the model on the input dataset.'''
        labels = dataset.labels.cpu().numpy().ravel() # Get the non-one-hot encoded labels from the dataset. 
        return balanced_accuracy_score(labels, self.predict(dataset))
    
    def scale(self, dataset, fit:bool=True):
        # Repeatedly scaling a Dataset causes problems, though I am not sure why. I would have thought that
        # subsequent applications of a scaler would have no effect. 
        assert not dataset.scaled, 'Classifier.scale: The dataset has already been scaled.'

        embeddings = dataset.embeddings.cpu().numpy()
        if fit: # If specified, fit the scaler first. 
            self.scaler.fit(embeddings)
        embeddings = self.scaler.transform(embeddings)
        dataset.embeddings = torch.FloatTensor(embeddings).to(DEVICE) 
        dataset.scaled = True

    def fit(self, datasets:tuple, epochs:int=10, lr:float=1e-8, batch_size:int=16, sampler=None, weight_loss:bool=False):

        assert datasets.test.scaled, 'Classifier.fit: The input test Dataset has not been scaled.' 
        assert datasets.train.scaled, 'Classifier.fit: The input train Dataset has not been scaled.'

        self.train() # Put the model in train mode.

        if weight_loss: 
            self.loss_func.fit(datasets.train) # Set the weights of the loss function.

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-8)
        best_epoch, best_model_weights = 0, copy.deepcopy(self.state_dict())

        self.metrics['train_loss'] += [np.nan]
        self.metrics['test_acc'] += [self.accuracy(datasets.test)]

        dataloader = DataLoader(dataset.train, batch_size=batch_size, sampler=sampler, shuffle=True if (sampler is None) else False)

        pbar = tqdm(total=epochs * len(dataloader), desc=f'Classifier.fit: Training classifier, epoch 0 out of {epochs}.') 
        for epoch in range(epochs):
            self.loss = list()
            for batch in dataloader:
                # Evaluate the model on the batch in the training dataloader. 
                outputs, targets = self(batch['embedding']), batch['label_one_hot_encoded'] 
                loss = self.loss_func(outputs, targets)
                loss.backward() 
                self.loss.append(loss.item()) # Store each loss for computing metrics. 
                
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1) # Update progress bar after each batch. 
            
            self.metrics['train_loss'] += [np.mean(self.loss)]
            self.metrics['test_acc'] += [self.accuracy(datasets.test)]

            pbar.set_description(f'Classifier.fit: Training classifier, epoch {epoch} out of {epochs}. test_acc={self.metrics['test_acc'][-1]}')

            if self.metrics['test_acc'][-1] > max(self.metrics['test_acc'][:-1]):
                best_epoch, best_model_weights = epoch, copy.deepcopy(self.state_dict())

        print(f'Classifier.fit: Loading best model weights from epoch {best_epoch}.')
        self.load_state_dict(best_model_weights) # Load the best model weights. 

        # Save training parameters in the model. 
        self.best_epoch = best_epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_loss = weight_loss


    @classmethod
    def load(cls, path:str):
        with open(path, 'rb') as f:
            # obj = pickle.load(f)
            obj = Unpickler(f).load()
        return obj.to(DEVICE)  

    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
