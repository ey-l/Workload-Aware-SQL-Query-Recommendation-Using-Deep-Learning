import sys
import os
import gc
import pprint
import yaml
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Imports based on the yaml configuration
script_dir = '/home/eugenie/projects/def-rachelpo/eugenie/queryteller/scripts/models/'
sys.path.append(script_dir)

'''
if sys.argv[1] == 'transformer':
    from transformer import core, training
elif sys.argv[1] == 'rnn':
    from rnn import core, training
else: from cnn import core, training
'''

from imports import *
from utils import * 
from dataloader import *
from eval import * 
from search import *
from seq_generator import *
from early_stopping import * 

if __name__ == "__main__":
    # Empty cache
    gc.collect()
    torch.cuda.empty_cache()

    # Use CUDA if it is available
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from cnn import core, training
    default_tokens = training.DEFAULT_TOKENS
    max_length = training.MAX_LENGTH

    # Load data
    data_dir = '/home/eugenie/projects/def-rachelpo/eugenie/data/processed/sdss/model_data/sampled/'
    save_dir = os.path.join("data", "save")
    datafile = os.path.join(data_dir, 'train.txt')
    voc, pairs = load_query_pairs(default_tokens, data_dir, 'sdss', datafile, save_dir, max_length)
    pairs = trimRareWords(voc, pairs, 3)
    datafile = os.path.join(data_dir, 'test.txt')
    _, test_data = load_query_pairs(default_tokens, data_dir, 'sdss', datafile, save_dir, max_length)
    test_data = trimRareWords(voc, test_data, 3)

    # Load CNN
    model_pth = '../../saved_models/sdss/cnn_pred_sdss.pth'
    loaded_model = core.make_model(voc.num_words, 100, 256, 6, 3, 3, 0.1, 0, device, 102).to(device)
    loaded_model.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu'))) # Load to cpu
    loaded_model.eval()
    print("Model loaded...")

    with torch.no_grad():
        start_time = time.time()
        test_size = len(test_data)
        data_iter = list(training.data_to_batch(voc, test_data, device, batch_size=1)) #len(pairs)

        # Use greedy decoding for set prediction
        trgs, preds = generate_topK_beams(data_iter, loaded_model, voc)
        # Get fragment dict from K beams
        trg_dicts, pred_dicts = get_fragment_dict(voc, trgs, preds, False)

        result = calc_f_measure(trg_dicts, pred_dicts, None)

    print(result)