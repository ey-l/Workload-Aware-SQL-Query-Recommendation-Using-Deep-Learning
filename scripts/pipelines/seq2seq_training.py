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

'''
Training pipeline functions, integrating wandb
'''

# Load yaml file with arg
setup_config = os.path.join('/home/eugenie/projects/def-rachelpo/eugenie/queryteller/scripts/configs/seq2seq/', sys.argv[1])
with open(setup_config) as f:
	setup_config = yaml.safe_load(f)
setup = setup_config['setup']
hp_range = setup_config['range']
hp_constant = setup_config['constant']
hp_constant['architecture']['value'] = setup['architecture']

# Load setup yaml file to update configuration
setup_config = os.path.join('/home/eugenie/projects/def-rachelpo/eugenie/queryteller/scripts/configs/seq2seq/setup.yaml')
with open(setup_config) as f:
	setup_config = yaml.safe_load(f)
setup.update(setup_config['setup'])
hp_constant.update(setup_config['constant'])

# Imports based on the yaml configuration
sys.path.append(setup['scripts_dir'])
if setup['architecture'] == 'transformer':
    from transformer import core, training
elif setup['architecture'] == 'rnn':
    from rnn import core, training
else: from cnn import core, training

#sys.path.append(os.path.join(setup['scripts_dir'], 'skyrec'))
from imports import *
from utils import * 
from dataloader import *
from eval import * 
from search import *
from seq_generator import *
from early_stopping import * 

def model_pipeline(config=None):

    # tell wandb to get started
    with wandb.init(config=config): #, project='queryrec21', entity='eylai'
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
            
        # load data
        train_data = random.sample(train_pairs, int(config.data_frac * len(train_pairs)))
        valid_data = random.sample(valid_pairs, int(config.data_frac * len(valid_pairs)))
        test_data = random.sample(test_pairs, int(config.data_frac * len(test_pairs)))

        # make the model, data, and optimization problem
        model, criterion, optimizer = training.make(config)
        #print(model)
        config.param_count = count_parameters(model)
        wandb.log({"param_count": config.param_count})

        # initialize the early_stopping object with a path for checkpoint
        early_stopping = EarlyStopping(patience=config.patience, verbose=True, path='/home/eugenie/scratch/exp/pipeline_checkpoint.pt')

        # and use them to train the model
        model = training.train(model, voc, train_data, valid_data, criterion, optimizer, early_stopping, config, train_log)

        # and test its final performance
        #test(model, test_data, config)

    return model

def train_log(train_loss, valid_loss, epoch_time, epoch):
    '''
    :param example_ct: number of examples seen
    '''

    # where the magic happens
    wandb.log({"epoch": epoch, "train_loss": float(train_loss), "valid_loss": float(valid_loss), "epoch_time": epoch_time})
    print(f"End of epoch " + str(epoch) + f" | train loss: {train_loss:.3f}" + f" | validation loss: {valid_loss:.3f}" + f" | epoch_time: {epoch_time}")

def test(model, test_data, config):
    model.eval()

    # Make beam search strategy 
    bs_strategy = BeamSearch(voc)

    # Run the model on some test examples
    with torch.no_grad():
        start_time = time.time()
        test_size = len(test_data)

        # Randomly sample 2000 query pairs
        if test_size > 2000:
            test_size = 2000
            test_data = random.sample(test_data, test_size)

        data_iter = list(training.data_to_batch(voc, test_data, device, batch_size=1))

        '''
        # Use beam search
        trgs, preds = generate_topK_beams(data_iter, model, voc, K=5, bs_strategy=bs_strategy) # only consider k=1
        # Get fragment dict from K beams
        trg_dicts, pred_dicts = get_fragment_dict(voc, trgs, preds, True)
        '''

        # Use greedy decoding for set prediction
        trgs, preds = generate_topK_beams(data_iter, model, voc)
        # Get fragment dict from K beams
        trg_dicts, pred_dicts = get_fragment_dict(voc, trgs, preds, False)

        # Get evalulation results, cares about the whole pred set when testing, instead of varying K
        # F-measure
        avg_f_set, _ = calc_f_measure(trg_dicts, pred_dicts) # N=None, considering the whole set

        print(f"Accuracy of the model on {test_size} " +
              f"query pairs | f-measure fragment: {100 * avg_f_set['fragments']:.2f}%" + 
              f"| f-measure table: {100 * avg_f_set['tables']:.2f}%" + 
              f"| f-measure attribute: {100 * avg_f_set['attributes']:.2f}%" + 
              f"| f-measure function: {100 * avg_f_set['functions']:.2f}%" + 
              f"| f-measure literal: {100 * avg_f_set['literals']:.2f}%")
        
        wandb.log({"test_f_fra": avg_f_set['fragments'], "test_f_tab": avg_f_set['tables'], "test_f_att": avg_f_set['attributes'], "test_f_fun": avg_f_set['functions'], "test_f_lit": avg_f_set['literals'], "test_time": time.time() - start_time, "test_size":test_size})
	
if __name__ == "__main__":
    # Empty cache
    gc.collect()
    torch.cuda.empty_cache()

	# Use CUDA if it is available
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp_constant['device']['value'] = device
    
    global default_tokens, max_length
    default_tokens = training.DEFAULT_TOKENS
    max_length = training.MAX_LENGTH

	# Load data
    global voc, train_pairs, valid_pairs, test_pairs
    voc, data = get_data(setup['data_dir'], os.path.join(setup['dataset'], setup['data_folder']), setup['if_aware'], default_tokens, max_length)
    train_pairs, valid_pairs, test_pairs = data
    hp_constant['V']['value'] = voc.num_words
    hp_constant['dataset']['value'] = setup['dataset']
    hp_constant['if_aware']['value'] = setup['if_aware']
    
    # Make config
    sweep_config = setup['sweep_config']
    sweep_config['parameters'] = hp_range
    hp_range.update(hp_constant)
    pprint.pprint(sweep_config)
    
    # Start sweeping
    sweep_id = wandb.sweep(sweep_config, project=setup['project'])
    wandb.agent(sweep_id, model_pipeline) # count=setup['runs']
    
    gc.collect()
    torch.cuda.empty_cache()
