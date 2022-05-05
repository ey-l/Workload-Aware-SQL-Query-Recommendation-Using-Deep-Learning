from imports import * 

'''
Core
'''
class QueryClassifier(nn.Module):
    def __init__(self, pretrained_encoder, architecture, num_classes, n1_size, n2_size, dropout=0.2):
        '''
        :param architecture (str): architecture of the encoder, either rnn, cnn, or transformer
        :param n1_size (int): the output size of the encoder, not tunable. CNN: 100
        :param n2_size (int): tunable HP
        '''
        super(QueryClassifier, self).__init__()
        self.encoder = pretrained_encoder
        self.encoder_arch = architecture
        self.fc1 = nn.Linear(n1_size, n2_size)
        self.fc2 = nn.Linear(n2_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, lengths=None):
        if self.encoder_arch == 'rnn': 
            _, x = self.encoder(x, mask, lengths)
            x = x[-1, :, :]
        elif self.encoder_arch == 'cnn': 
            x, _ = self.encoder(x)
            x = x[:, 0, :]
        else:
            x = self.encoder(x, mask)
            x = x[:, 0, :]

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def make_classifier(encoder, architecture, num_classes, n1_size, n2_size, dropout):
    return QueryClassifier(encoder, architecture, num_classes, n1_size, n2_size, dropout)

class NaiveClassifier(nn.Module):
    def __init__(self, device, V, emb_size, num_classes, hidden_size, batch_size, dropout=0.0):
        super(NaiveClassifier, self).__init__()
        self.device = device
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.embed = nn.Embedding(V, emb_size)
        self.fc1 = nn.Linear(emb_size * 100, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print(x.shape)
        x = x.long()
        x = self.embed(x)
        x = torch.flatten(x, start_dim=1)
        padding = torch.zeros(self.batch_size, self.emb_size*100).to(self.device)
        padding[:,:x.shape[1]] = x
        x = padding
        #print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def make_naive_classifier(device, V, num_classes, batch_size):
    return NaiveClassifier(device, V, 100, num_classes, hidden_size=512, batch_size=batch_size)

'''
Data
'''
class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0):
        
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = trg
        
        if torch.cuda.is_available():
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()
            self.trg = self.trg.cuda()

def vector_from_label(l):
    vector = [0] * len(templates)
    vector[l] = 1
    return vector

'''
Decoding
'''
def pred_class(pairs, model, classifier, voc):
    '''
    Return the index representation and the vector representation of the query statements
    :param model (nn.Module): trained seq2seq model
    :param classifier (nn.Module): classification model

    :return src: numpy array
    :return vectors: numpy array
    '''
    src = []
    eval_data = []
    for pair in pairs:
        eval_data.append([pair[0], 0])
    
    data = queries_to_batch(voc, eval_data)
    for i, batch in enumerate(data, 1):
        # One-hot encoding of the query statements
        src = batch.src.cpu().detach().numpy()
        # Get the vector representation
        _, final = model.encoder(model.src_embed(batch.src), batch.src_mask, batch.src_lengths)
        outputs = classifier(model.src_embed(batch.src), batch.src_mask, batch.src_lengths)
        _, outputs = torch.max(outputs, 1)

    return outputs
