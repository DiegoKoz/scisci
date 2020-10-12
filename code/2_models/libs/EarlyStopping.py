import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss/auc doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, metric = 'loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            metric(string): wether to implement the early stopping using the loss or the AU'{'loss','auc'}
                            Default: 'loss'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.value_min = np.Inf
        self.delta = delta
        self.metric = metric

    def __call__(self, value, model):
        
        if self.metric == 'loss':
            score = -value
        elif self.metric == 'auc':
            score = value

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(value, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(value, model)
            self.counter = 0

    def save_checkpoint(self, value, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.value_min:.6f} --> {value:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.value_min = value