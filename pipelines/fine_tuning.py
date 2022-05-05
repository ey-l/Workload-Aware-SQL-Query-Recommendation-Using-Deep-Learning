import sys
import os
import gc
import pprint
import yaml
import wandb
os.environ["WANDB_DIR"] = '/home/eugenie/projects/def-rachelpo/eugenie/' # Set wandb dir
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load setup yaml file to update configuration
setup_config = os.path.join('/home/eugenie/projects/def-rachelpo/eugenie/queryteller/scripts/configs/classification/setup.yaml')
with open(setup_config) as f:
    setup_config = yaml.safe_load(f)
setup = setup_config['setup']
hp_range = setup_config['range']
hp_constant = setup_config['constant']

# Load yaml file with arg
setup_config = os.path.join('/home/eugenie/projects/def-rachelpo/eugenie/queryteller/scripts/configs/classification/', sys.argv[1])
with open(setup_config) as f:
    setup_config = yaml.safe_load(f)
setup.update(setup_config['setup'])
model_config = setup_config['constant']
hp_constant['emb_size']['value'] = model_config['output_size']
hp_constant['architecture']['value'] = setup['architecture']

# Imports based on the yaml configuration 
sys.path.append(setup['scripts_dir'])
if setup['architecture'] == 'transformer':
    from transformer import core, training
elif setup['architecture'] == 'rnn':
    from rnn import core, training
else: from cnn import core, training

'''
Training pipeline functions, integrating wandb
'''

sys.path.append(setup['scripts_dir'])
from imports import *
from utils import * 
from dataloader import *
from eval import * 
from early_stopping import *
from template import *
from classification import *

def model_pipeline(config=None):
    
    # tell wandb to get started
    with wandb.init(config=config): #, project='queryrec21', entity='eylai'
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        config.device = device
        
        pprint.pprint(config)
        
        # load data
        train_data = random.sample(train_pairs, int(config.data_frac * len(train_pairs)))
        valid_data = random.sample(valid_pairs, int(config.data_frac * len(valid_pairs)))
        test_data = random.sample(test_pairs, int(config.data_frac * len(test_pairs)))

        # make the model, data, and optimization problem
        model, criterion, optimizer = make(config)
        config.param_count = count_parameters(model)

        # initialize the early_stopping object with a path for checkpoint
        early_stopping = EarlyStopping(patience=config.patience, verbose=True, path='/home/eugenie/scratch/exp/'+config.architecture+'_checkpoint.pt')

        # and use them to train the model
        model = train(model, voc, train_data, valid_data, criterion, optimizer, early_stopping, config, train_log)

        # and test its final performance
        test(model, test_data, config)
        
    return model

def make(config):
    encoder =  loaded_model.encoder
    model = make_classifier(encoder, config.architecture, config.V, config.emb_size, config.hidden_size, config.dropout).to(config.device[0])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    return model, criterion, optimizer

def train(model, voc, train_data, valid_data, criterion, optimizer, early_stopping, config, train_log): 
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()

        data = data_to_batch(voc, train_data, batch_size=config.batch_size)
        train_loss = 0.0
        norm = 0

        for i, batch in enumerate(data, 1):

            model.train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if config.architecture == 'rnn': 
                outputs = model(loaded_model.src_embed(batch.src), batch.src_mask, batch.src_lengths)
            elif config.architecture == 'cnn': 
                outputs = model(batch.src, batch.src_mask)
            else:
                outputs = model(loaded_model.src_embed(batch.src), batch.src_mask)
            loss = criterion(outputs, batch.trg)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            norm = i+1
        train_loss = train_loss / norm
        
        eval_data = data_to_batch(voc, valid_data, batch_size=config.batch_size)
        valid_loss = 0.0
        norm = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(eval_data, 1):
                # forward + backward + optimize
                if config.architecture == 'rnn': 
                    outputs = model(loaded_model.src_embed(batch.src), batch.src_mask, batch.src_lengths)
                elif config.architecture == 'cnn': 
                    outputs = model(batch.src, batch.src_mask)
                else:
                    outputs = model(loaded_model.src_embed(batch.src), batch.src_mask)
                loss = criterion(outputs, batch.trg)
                valid_loss += loss.item()
                norm = i + 1
        valid_loss = valid_loss / norm

        # Log training/validation loss
        epoch_time = time.time() - epoch_start_time
        train_log(train_loss, valid_loss, epoch_time, epoch)

        # Early_stopping needs the validation loss to check if it has decresed, 
        # And if it has, it will make a checkpoint of the current model
        early_stopping(train_loss, valid_loss, 0, model, on_loss=True)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    print('Finished Training')
    return model

def test(model, test_data, config):
    model.eval()
    N = 16
    test_size = len(test_data)
    test_data = data_to_batch(voc, test_data, batch_size=1)
    results = np.empty((test_size,N,))
    results.fill(0)

    # Run the model on some test examples
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            start_time = time.time()

            if config.architecture == 'rnn': 
                outputs = model(loaded_model.src_embed(batch.src), batch.src_mask, batch.src_lengths)
            elif config.architecture == 'cnn': 
                outputs = model(batch.src, batch.src_mask)
            else:
                outputs = model(loaded_model.src_embed(batch.src), batch.src_mask)

            # Get predicted and target templates
            _, preds = torch.topk(outputs, N, dim=1)
            preds = preds.cpu().reshape(-1).numpy()
            trg = batch.trg.cpu().numpy()

            # Set result to 1 if hits
            results[i,:] = calc_accuracy(trg, preds, N)

    avg_accuracy = mean_confidence_interval(results[:,0])[0] # get the m (i.e., [0]) of k=1 (i.e., [:,0])
    wandb.log({"test_accuracy": avg_accuracy, "test_time": time.time() - start_time, "test_size":test_size})
    
# Load the trained model
def load_seq_model(setup, config):
    if setup['architecture'] == 'transformer':
        loaded_model = core.make_model(config['V'], config['V'], N=config['num_layers'], d_model=config['emb_size'], d_ff=config['hidden_size'], h=config['heads'], dropout=config['dropout']).to(setup['device'])
    elif setup['architecture'] == 'rnn':
        loaded_model = core.make_model(config['V'], config['V'], emb_size=config['emb_size'], hidden_size=config['hidden_size'], num_layers=config['num_layers'], dropout=config['dropout'])
    else:
        enc = core.Encoder(config['V'], config['emb_size'], config['hidden_size'], config['num_layers'], config['kenel'], config['dropout'], setup['device'])
        dec = core.Decoder(config['V'], config['emb_size'], config['hidden_size'], config['num_layers'], config['kenel'], config['dropout'], 0, setup['device'])
        loaded_model = core.CNNSeq2Seq(enc, dec).to(setup['device'])
    
    model_pth = '/home/eugenie/projects/def-rachelpo/eugenie/queryteller/saved_models_v2/'+setup['architecture']+'_'+setup['dataset']+'_'+str(setup['if_aware'])+'_checkpoint.pt'
    loaded_model.load_state_dict(torch.load(model_pth)) # Load to cpu
    loaded_model.eval()

    return loaded_model

def data_to_batch(voc, pairs, device='cpu', batch_size=64, pad_index=0, sos_index=1):
    # Load batches for each iteration
    num_batches = int(len(pairs) / batch_size)
    batches = [put_data_in_batch(voc, random.sample(pairs, batch_size))
                      for _ in range(num_batches)]
    for i in range(num_batches):
        batch = batches[i]
        src, src_lengths, trg = batch
        src = Variable(src, requires_grad=False).to(device) 
        #src_lengths = Variable(src_lengths, requires_grad=False).to(device)
        trg = Variable(trg, requires_grad=False).to(device)
        #trg_lengths = Variable(trg_lengths, requires_grad=False).to(device)
        yield Batch((src, src_lengths), trg, pad_index=pad_index)

def train_log(train_loss, valid_loss, epoch_time, epoch):
    '''
    :param example_ct: number of examples seen
    '''

    # where the magic happens
    wandb.log({"epoch": epoch, "train_loss": float(train_loss), "valid_loss": float(valid_loss), "epoch_time": epoch_time})
    print(f"End of epoch " + str(epoch) + f" | train loss: {train_loss:.3f}" + f" | validation loss: {valid_loss:.3f}" + f" | epoch_time: {epoch_time}")

if __name__ == "__main__":

    # Use CUDA if it is available
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup['device'] = device
    hp_constant['device']['value'] = device

    global default_tokens, max_length
    default_tokens = training.DEFAULT_TOKENS
    max_length = training.MAX_LENGTH
    
    # Load data
    global voc, train_pairs, valid_pairs, test_pairs
    voc, _ = get_data(setup['data_dir'], os.path.join(setup['dataset'], setup['seq_data']), setup['if_aware'], default_tokens, max_length)
    _, data = get_data(setup['data_dir'], os.path.join(setup['dataset'], setup['class_data']), setup['if_aware'], default_tokens, max_length, voc)
    train_pairs, valid_pairs, test_pairs = data
    hp_constant['dataset']['value'] = setup['dataset']
    hp_constant['if_aware']['value'] = setup['if_aware']
    hp_constant['V']['value'] = voc.num_words
    model_config['V'] = voc.num_words
    
    # Get template list from training data
    train_pairs, templates, template_counter = extract_templates(voc, train_pairs, setup['template_min_count'])
    train_pairs = trim_pairs(voc, templates, train_pairs)
    valid_pairs = trim_pairs(voc, templates, valid_pairs)
    test_pairs = trim_pairs(voc, templates, test_pairs)
    
    # Load trained seq2seq model
    loaded_model = load_seq_model(setup, model_config)

    wandb.agent(sys.argv[2], model_pipeline) # count=setup['runs']

    #model_pipeline(config)
    
    # Start sweeping
    #sweep_id = wandb.sweep(sweep_config, project=setup['project'])
    #wandb.agent(sweep_id, model_pipeline) # count=setup['runs']

