from io import open
from imports import *
import torch
import itertools

'''
Functions shared among models
'''

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

#KEYWORDS = '(select|insert|delete|update|upsert|replace|merge|drop|create|alter|where|from|inner|join|straight_join|and|like|set|by|group|order|left|outer|full|if|end|then|loop|else|for|while|case|when|min|max|distinct)'

MAX_LENGTH = 300
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

########################
######  Evaluate  ######
########################

def sort_dict(d, reverse=False):
    '''
    Sort dictionary based on value, return a dict
    '''
    sorted_dict = OrderedDict(sorted(d.items(), key=operator.itemgetter(1), reverse=reverse))
    return sorted_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]

def print_examples(example_iter, model, voc, n=2, max_len=100, 
                   sos_index=1, 
                   src_eos_index=None, 
                   trg_eos_index=None, 
                   src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()
    
    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = src_vocab.stoi[EOS_TOKEN]
        trg_sos_index = trg_vocab.stoi[SOS_TOKEN]
        trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
    else:
        src_eos_index = 2
        trg_sos_index = 1
        trg_eos_index = 2
        
    for i, batch in enumerate(example_iter):
      
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg      
      
        result, _ = greedy_decode(
          model, batch.src, batch.src_mask, batch.src_lengths,
          max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i+1))
        print("Input : ", " ".join([voc.index2word[token.item()] for token in src]))
        print("Target : ", " ".join([voc.index2word[token.item()] for token in trg]))
        print("Pred: ", " ".join([voc.index2word[token.item()] for token in result]))
        print()
        
        count += 1
        if count == n:
            break

######################
#######  Data  #######
######################

def print_queries(file, n=10):
    '''
    Print a sample of queries
    :param file: the file path we want to sample
    '''
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

def indexesFromSentence(voc, sentence):
    '''
    Translate each word in a query to its index number in the built vocabulary
    :return : a list of corresponding indexes of the query
    '''
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    '''
    Pad the vectors with 0's
    '''
    return list(itertools.zip_longest(*l, fillvalue=fillvalue)) # *l transpose the matrix

def binaryMatrix(l, value=PAD_token):
    '''
    Return a matrix of 1's and 0's. If a entry is non-zero, the corresponding entry is 1
    '''
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l, voc):
    '''
    Converting sentences to tensor, ultimately creating a correctly shaped zero-padded tensor
    :return padVar: padded input sequence tensor
    :return lengths: the length
    '''
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    '''
    Returns padded target sequence tensor, padding mask, and max target length
    '''
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):
    '''
    Take query pairs and returns the input and target tensors using the aforementioned functions
    '''
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len
