from imports import *

'''
Early stopping class
'''

class EarlyStopping:
    '''
    Early stops the training if validation loss doesn't improve after a given patience.
    Credit: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py 
    '''
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        '''
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_accu_min = 0 # Added for accuracy measure
        self.train_loss = None
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, train_loss, val_loss, val_accu, model, on_loss=False):
        '''
        Track losses and validation accuracy.
        :param train_loss (float): train loss
        :param valid_loss (float): validation loss
        :param valid_accu (float): validation accuracy
        :param model (NN.Model): model in training
        :param on_loss (boolean): True when stop based on validation accuracy

        Use: 
            early_stopping(train_loss.item(), valid_loss.item(), avg_f, model)
        '''

        if on_loss:
            score = -val_loss
        else:
            score = val_accu

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_loss, val_loss, val_accu, model, on_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss, val_loss, val_accu, model, on_loss)
            self.counter = 0

    def save_checkpoint(self, train_loss, val_loss, val_accu, model, on_loss):
        '''
        Saves model when validation loss decreases or validation accuracy increases.
        :param train_loss (float): train loss
        :param valid_loss (float): validation loss
        :param valid_accu (float): validation accuracy
        :param model (NN.Model): model in training
        :param on_loss (boolean): True when stop based on validation accuracy
        '''
        if self.verbose:
            if on_loss:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                self.trace_func(f'Validation accuracy increased ({self.val_accu_min:.6f} --> {val_accu:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.val_accu_min = val_accu
        self.train_loss = train_loss
