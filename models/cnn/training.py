import sys
sys.path.append('../')
from imports import *
from dataloader import *
from cnn import core

DEFAULT_TOKENS = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3, 'SEM': 4}
MAX_LENGTH = 100

def make(config):
    # Criterion
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # Make model
    enc = core.Encoder(config.V, config.emb_size, config.hidden_size, config.num_layers, config.kernel, config.dropout, config.device[0], 102)
    dec = core.Decoder(config.V, config.emb_size, config.hidden_size, config.num_layers, config.kernel, config.dropout, 0, config.device[0], 102)
    model = core.CNNSeq2Seq(enc, dec).to(config.device[0]) 

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
        train_loss = run_epoch(data, model, criterion, optimizer)
        
        # Validation
        model.eval()
        with torch.no_grad():
            # Get valid data to Batch
            eval_data = data_to_batch(voc, valid_data, config.device[0], config.batch_size)
            valid_loss = run_epoch(eval_data, model, criterion, None)
        
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

def run_epoch(data_iter, model, criterion, optimizer, clip=0.1):
    """Standard Training and Logging Function"""
    total_loss = 0
    norm = 0

    for i, batch in enumerate(data_iter, 1):
        
        output, _ = model(batch.src, batch.trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = batch.trg[:,1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        
        if optimizer is not None:
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        # Get the batch size
        norm = i

    return total_loss / norm

'''
Data
'''
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

# Change to adapt to transformer
def data_to_batch(voc, pairs, device='cpu', batch_size=64, pad_index=0, sos_index=1):
    # Load batches for each iteration
    num_batches = int(len(pairs) / batch_size)
    batches = [put_data_in_batch(voc, random.sample(pairs, batch_size))
                      for _ in range(num_batches)]
    for i in range(num_batches):
        batch = batches[i]
        src, src_lengths, trg, trg_lengths = batch
        src = Variable(src, requires_grad=False).to(device)
        trg = Variable(trg, requires_grad=False).to(device)
        yield Batch(src, trg, pad_index)
