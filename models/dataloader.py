import sys
sys.path.append('./')
from imports import *

class Voc:
    def __init__(self, name, default_tokens):
        self.name = name
        self.trimmed = False
        self.init_dict = default_tokens # {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3, 'SEM': 4}

        # Swap the k:v pairs. E.g., {0: "PAD", 1: "SOS", 2: "EOS", 3: 'UNK', 4: 'SEM'}
        self.index2word = dict((v,k) for k,v in self.init_dict.items())
        self.num_words = len(self.init_dict.keys())
        self.word2index = {}
        self.word2count = {}

        self.pad = self.init_dict['PAD'] # Used for padding short sentences
        self.sos = self.init_dict['SOS'] # Start-of-sentence token
        self.eos = self.init_dict['EOS'] # End-of-sentence token
        self.unk = self.init_dict['UNK'] # Unknown voc token
        self.sem = None # Query-level semantic token; added for sentence-level transformer and cnn

        if 'SEM' in self.init_dict:
            self.sem = self.init_dict['SEM']

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word, count=None):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1
            if count is None:
                self.word2count[word] = 1
            else:
                self.word2count[word] = count
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append((k, v))

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = dict((v,k) for k,v in self.init_dict.items())
        self.num_words = len(self.init_dict.keys())

        for word, count in keep_words:
            self.addWord(word, count)

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name, default_tokens):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[s for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name, default_tokens)
    return voc, pairs

# Returns True iff both sentences in a pair 'p' are under the max_length threshold
def filterPair(p, max_length):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length

# Filter pairs using filterPair condition
def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]

def load_query_pairs(default_tokens, corpus, corpus_name, datafile, save_dir, max_length):
    '''
    Using the functions defined above, return a populated voc object and pairs list
    :param default_tokens (dict): dict of default tokens
    e.g., {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3, 'SEM': 4}

    '''
    print("Start preparing data ...")
    voc, pairs = readVocs(datafile, corpus_name, default_tokens)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

def swap_unk_in_query(voc, s):
    # Check input sentence
    s = s.split(' ')
    for w in range(len(s)):
        if s[w] not in voc.word2index:
            # Set to UNK
            s[w] = 'UNK'
    return ' '.join(s)

def trimRareWords(voc, pairs, min_n=3):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(min_n)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        pair[0] = swap_unk_in_query(voc, pair[0])
        pair[1] = swap_unk_in_query(voc, pair[1])
    return pairs

def print_lines(file, n=10):
    '''
    Print a sample of lines
    :param file: the file path we want to sample
    '''
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

def make_pairs_with_single(filepath):
    q_pairs = []
    with open(filepath, 'r', errors='ignore') as f:
        reader = csv.DictReader(f)
        next(reader) # Skip header
        for row in reader:
            sqlstatement = row['statement'].replace('\t', ' ').replace('..', ' ')
            q_pairs.append([sqlstatement, sqlstatement])
    return q_pairs

def make_pairs(filepath, datafile, lineterminator, delimiter, n=10):
    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator=lineterminator)
        for pair in make_pairs_with_single(filepath):
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from file:")
    print_lines(datafile, n)

def indexesFromSentence(voc, sentence):
    '''
    Translate each word in a query to its index number in the built vocabulary
    :return : a list of corresponding indexes of the query
    '''
    indexes = []
    for word in sentence.split(' '):
        if word != 'UNK':
            indexes += [voc.word2index[word]]
        else: indexes += [voc.unk]
    if voc.sem is not None:
        return [voc.sos] + [voc.sem] + indexes + [voc.eos] # Added 'SEM'
    else:
        return [voc.sos] + indexes + [voc.eos]

def zeroPadding(l, fillvalue=0):
    '''
    Pad the vectors with 0's
    '''
    return np.array(list(itertools.zip_longest(*l, fillvalue=fillvalue))).T # *l transpose the matrix

def prepare_input_tensor(l, voc):
    '''
    Converting sentences to tensor, ultimately creating a correctly shaped zero-padded tensor
    :return padVar: padded input sequence tensor
    :return lengths: the length
    '''
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = [len(indexes)-1 for indexes in indexes_batch]
    lengths = torch.tensor(np.array(lengths))
    indexes_batch = zeroPadding(indexes_batch, voc.pad)
    indexes_batch = torch.tensor(indexes_batch).to(torch.int64)
    indexes_batch = indexes_batch[:, 1:]
    return indexes_batch, lengths

def prepare_output_tensor(l, voc):
    '''
    Converting sentences to tensor, ultimately creating a correctly shaped zero-padded tensor
    :return padVar: padded input sequence tensor
    :return lengths: the length
    '''
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = [len(indexes) for indexes in indexes_batch]
    lengths = torch.tensor(np.array(lengths))
    indexes_batch = zeroPadding(indexes_batch)
    indexes_batch = torch.tensor(indexes_batch).to(torch.int64)
    return indexes_batch, lengths

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
    trg, trg_lengths = prepare_output_tensor(output_batch, voc)
    return src, src_lengths, trg, trg_lengths

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def get_data(data_dir, folder, if_aware, default_tokens, max_length, voc=None, MIN_COUNT=3):
    '''
    :return voc (Voc): vocabulary built from train pairs
    :return data (list): [train, valid, test]
    '''
    data = []
    save_dir = os.path.join("data", "save")
    filenames = ['train', 'val', 'test']
    data_dir = os.path.join(data_dir, folder)
    for f in filenames:
        if (not if_aware) and (f in ['train', 'val']):
            f = ''.join([f,'_rec'])
        f = ''.join([f,'.txt'])
        datafile = os.path.join(data_dir, f)
        #print(f)
        if 'train' in f: 
            loaded_voc, pairs = load_query_pairs(default_tokens, data_dir, folder, datafile, save_dir, max_length)
        else:
            _, pairs = load_query_pairs(default_tokens, data_dir, folder, datafile, save_dir, max_length)
        
        if voc is None:
            voc = loaded_voc
        
        pairs = trimRareWords(voc, pairs, MIN_COUNT)
        data.append(pairs)
        print('-'*90)
    
    return voc, data

def split_query(q):
    '''
    Handling literals. Skip quotation marks
    :return (list): 
    '''
    return re.split(''' (?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', q)

if __name__ == "__main__":
    # Sample inputs
    default_tokens = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3, 'SEM': 4}
    max_length = 100
    data_dir = 'F://data//processed//sdss//model_data//sampled//'
    save_dir = os.path.join("data", "save")

    # Load training data
    datafile = os.path.join(data_dir, 'train.txt')
    voc, pairs = load_query_pairs(default_tokens, data_dir, 'sdss', datafile, save_dir, max_length)
    pairs = trimRareWords(voc, pairs, 3)

    # Load test data
    datafile = os.path.join(data_dir, 'test.txt')
    _, pairs = load_query_pairs(default_tokens, data_dir, 'sdss', datafile, save_dir, max_length)
    pairs = trimRareWords(voc, pairs, 3)