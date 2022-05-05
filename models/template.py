import sys
sys.path.append('./')
from imports import *
from utils import * 
from dataloader import *
from eval import * 
from early_stopping import * 

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        src, src_lengths = src

        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        if trg is not None:
            self.trg = trg

# Depends on the data.py of the imported seq2seq model
def trim_rare_words(voc, pairs, min_n=3):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(min_n)
    # Filter out pairs with trimmed words
    tlist = []
    for pair in pairs:
        pair[0] = swap_unk_in_query(voc, pair[0])
        temp = pair[1]
        tlist.append(temp)
        
    return pairs, tlist

# Trim voc and pairs
def trim_pairs(voc, templates, pairs, min_n=3):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(min_n)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        pair[0] = swap_unk_in_query(voc, pair[0])
        temp = pair[1]
        if temp in templates and 'ERROR' not in temp:
            pair[1] = templates.index(temp)
            keep_pairs.append(pair)
        
    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def put_data_in_batch(voc, pair_batch):
    '''
    Take query pairs and returns the input and target tensors using the aforementioned functions
    '''
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    src, src_lengths = prepare_input_tensor(input_batch, voc)
    trg = torch.tensor(output_batch)
    return src, src_lengths, trg

def extract_templates(voc, pairs, min_count=3):
    '''
    Get template list from training data
    :param voc (Voc): vocabulary built from seq2seq training data
    :param pairs (list): train_pairs
    :param min_count (int): template min count 

    :return pairs (list):
    :return templates (list): a list of unique templates
    '''
    # Get template list from training data
    pairs, tlist = trim_rare_words(voc, pairs)
    # Make a list of unique templates
    counter = Counter(tlist)
    tc = np.fromiter(counter.values(), dtype=int)
    filtered_counter = {k:v for k,v in dict(counter).items() if v > min_count} # 9 for sdss; 3 for sqlshr
    templates = list(filtered_counter.keys())

    return pairs, templates, tc

# Put pairs into a batch to feed into model
def queries_to_batch(voc, pairs):
    src, src_lengths, trg = put_data_in_batch(voc, pairs)
    src = Variable(src, requires_grad=False).to(device) 
    yield Batch((src, src_lengths), trg)

@torch.no_grad()
def query_to_vector(data, model):
    '''
    Return the index representation and the vector representation of the query statements
    :return data: 
    :return model: 
    '''
    trgs = None
    vectors = []
    for i, batch in enumerate(data, 1):
        trgs = batch.trg.cpu().detach().numpy()
        # Get the vector representation
        vectors = model.get_seq_vector(batch).cpu().detach().numpy()
    return trgs, vectors

def indexes_to_sentences(voc, src):
    '''
    Return the query statements of the input index representation
    :param voc: vocabulary
    :param src: numpy array
    :return statements: list of strings
    '''
    statements = []
    for x in q_src:
        s = " ".join([voc.index2word[token.item()] for token in x if (token != 0 and token != 2)])
        statements.append(s)
    return statements

def get_unique_pairs(pairs):
    '''
    Args:
        pairs (list): a list of lists
    '''
    unique_pairs = set(map(tuple, pairs))
    unique_pairs = list(map(list, unique_pairs))
    return unique_pairs