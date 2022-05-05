import sys
sys.path.append('../')
from imports import *
from dataloader import *
from rnn import core

DEFAULT_TOKENS = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
MAX_LENGTH = 100

def make(config):
    # Criterion
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    # Make model
    model = core.make_model(config.V, config.V, emb_size=config.emb_size, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout).to(config.device[0])
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    return model, criterion, optimizer

def train(model, voc, train_data, valid_data, criterion, optimizer, early_stopping, config, train_log):
    '''
    Train the model
    :param voc (Voc): vocabulary
    :param train_data (list): train data
    :param valid_data (list): validation data
    :param criterion:
    :param optimizer:
    :param early_stopping (EarlyStopping):
    :param config:

    :return model (nn.Module): best model based on validation loss
    '''

    for epoch in range(1, config.epochs+1):
        epoch_start_time = time.time()
        # Training
        model.train()
        # Get train data to Batch
        data = data_to_batch(voc, train_data, config.device[0], config.batch_size)
        #data = data_loader(voc, train_data, device, batch_size)
        train_loss = run_epoch(data, model, 
                  SimpleLossCompute(model.generator, criterion, optimizer))
        
        # Validation
        model.eval()
        with torch.no_grad():
            # Get valid data to Batch
            eval_data = data_to_batch(voc, valid_data, config.device[0], config.batch_size)
            #eval_data = data_loader(voc, valid_data, device, batch_size)
            valid_loss = run_epoch(eval_data, model, 
                            SimpleLossCompute(model.generator, criterion, None))
        
        # Log training/validation loss
        epoch_time = time.time() - epoch_start_time
        train_log(train_loss, valid_loss, epoch_time, epoch)

        # Early_stopping needs the validation loss to check if it has decresed, 
        # And if it has, it will make a checkpoint of the current model
        early_stopping(train_loss, valid_loss, 0, model, on_loss=True)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(early_stopping.path))

    print('Finished Training')
    return model

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm

def run_epoch(data_iter, model, loss_compute, print_every=50):
    """Standard Training and Logging Function"""

    total_tokens = 0
    total_loss = 0

    for i, batch in enumerate(data_iter, 1):
        
        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss = loss_compute(pre_output, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens

    return total_loss / total_tokens

'''
Data
'''
class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg=None, pad=0):
        
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad)
            self.ntokens = (self.trg_y != pad).data.sum().item()
        
        if torch.cuda.is_available():
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()

def data_to_batch(voc, pairs, device='cpu', batch_size=64, pad_index=0, sos_index=1):
    # Load batches for each iteration
    num_batches = int(len(pairs) / batch_size)
    batches = [put_data_in_batch(voc, random.sample(pairs, batch_size))
                      for _ in range(num_batches)]
    for i in range(num_batches):
        batch = batches[i]
        src, src_lengths, trg, trg_lengths = batch
        src = Variable(src, requires_grad=False).to(device) 
        #src_lengths = Variable(src_lengths, requires_grad=False).to(device)
        trg = Variable(trg, requires_grad=False).to(device)
        #trg_lengths = Variable(trg_lengths, requires_grad=False).to(device)
        yield Batch((src, src_lengths), (trg, trg_lengths), pad=pad_index)
