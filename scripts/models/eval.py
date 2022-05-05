import sys
from imports import *
from sqlparse import keywords, tokens

# Import TABLES and VIEWS for SDSS
from skyrec import options 
from utils import *

'''
Evaluation functions
'''

TOKENS = ['num', 'unk', 'eos', 'sos', 'sem', 'id']
F_TYPES = ['fragments', 'tables', 'attributes', 'functions', 'literals']
# For SDSS datasets
mergedList = options.TABLES + options.VIEWS
mergedList = [t.lower() for t in mergedList]
functionList = [f.lower() for f in options.FUNCTIONS]

def calc_accuracy(trg, preds, N=1):
    '''
    Calculate accracy for template prediction, per query
    Args:
        trg (int): the target class of the query
        preds (list): the list of predicted classes 
    '''
    results = np.empty((N,))
    results.fill(0)

    # Set result to 1 if hits
    for n in range(N):
        pred = preds[:n+1]
        if trg in pred:
            results[n] = 1

    return results

def calc_mrr(trg, preds, N=1):
    '''
    Calculate MRR for template prediction, per query
    Args:
        trg (int): the target class of the query
        preds (list): the list of predicted classes 
    '''
    results = np.empty((N,))
    results.fill(0)
    rank = 0

    # Set rank if hits
    for n in range(N):
        pred = preds[n]
        if trg == pred:
            rank = 1 / (n+1)
            results[n] = rank
        else:
            results[n] = rank

    return results

def calc_ndcg(trg, preds, N=1):
    '''
    Calculate NDCG for template prediction, per query
    Args:
        trg (int): the target class of the query
        preds (list): the list of predicted classes 
    '''
    results = np.empty((N,))
    results.fill(0)
    rank = 0

    # Set rank if hits
    for n in range(N):
        pred = preds[n]
        if trg == pred:
            rank = 1 / math.log((n+1)+1, 2) # log base 2
            results[n] = rank
        else:
            results[n] = rank

    return results

def calc_f_measure(trgs, preds, N=None):
    '''
    Calculate a given metric, either precision or recall.
    :param trg: a list of lists of tokens, i.e., 2d array
    :param pred: a list of lists (i.e., top k) of lists of tokens, i.e., 3d array. Same length as trg
    '''
    if N is not None:
        topN = N
    else: topN = 1
    
    # Precision
    prec_dict = calc_fragment_metric(trgs, preds, N, precision)
    # Recall
    recall_dict = calc_fragment_metric(trgs, preds, N, recall)
    # F-measure
    avg_f_set = {}
    avg_f = {}
    for t in F_TYPES:
        # Only takes the first col. Used for eval the whole pred set
        avg_f_set[t] = mean_confidence_interval(f_measure(prec_dict[t], recall_dict[t])[:,0])[0] # get the m (i.e., [0]) of k=1 (i.e., [:,0])
        
        # Used when eval for K
        for n in range(topN):
            f_m = mean_confidence_interval(f_measure(prec_dict[t], recall_dict[t])[:,n])[0] # get the m (i.e., [0]) of k=1 (i.e., [:,0])
            if t not in list(avg_f.keys()):
                avg_f[t] = []
            avg_f[t].append(f_m)

    return avg_f_set, avg_f

def is_keyword(t):
    '''
    Check if the token is a keyword using the Token.Keyword lists of sqlparse package
    :param t: a string containing a word token
    :return: boolean, true if t is a keyword
    '''
    token_type = keywords.is_keyword(t)[0]
    return (token_type == tokens.Keyword) or (token_type == tokens.DDL) or (token_type == tokens.DML) or (token_type == tokens.CTE)

def is_word_token(t):
    '''
    Check if a token is a special char
    :param t: a string containing a special char
    :return: boolean, false if t only contains special chars
    '''
    return not re.match(r'^[_\W]+$', t)

def get_fragments(tokens, lookup_dict=None):
    '''
    Extract each type of fragments from a list of word tokens
    :param tokens: a list of strings, where each string is a word token
    :param lookup_dict (dict): for looking up fragments of each type, used for SQLShare

    :return fragment_dict: a dictionary
    '''
    # Make a dict
    fragment_dict = {
        "fragments": [],
        "functions": [],
        "literals": [],
        "tables": [],
        "attributes": []
    }

    # Remove special chars
    tokens =  [str(t).lower() for t in tokens if is_word_token(t)]
    # Remove duplicate tokens
    tokens = list(set(tokens))
    # Get fragments, excluding keywords and model tokens
    fragment_dict['fragments'] = [t for t in tokens if not (is_keyword(t) or (t in TOKENS) or t == 'top')] #or (t in fragment_dict['literals'])
    # Get literals
    fragment_dict['literals'] = [t for t in tokens if '\'' in t]
    
    if lookup_dict is None:
        # Get tables, attributes, and functions, SDSS specific
        fragment_dict['tables'] = [t for t in fragment_dict['fragments'] if t in mergedList]
        fragment_dict['functions'] = [t for t in fragment_dict['fragments'] if (t in functionList) | ('dbo.' in t)]
        fragment_dict['attributes'] = [t for t in fragment_dict['fragments'] if (t not in fragment_dict['tables']) & (t not in fragment_dict['functions']) & (t not in fragment_dict['literals'])]
    else:
        # Get fragments by type
        fragment_dict['functions'] = [t for t in tokens if t in lookup_dict['functions']]
        fragment_dict['attributes'] = [t for t in tokens if t in lookup_dict['attributes']]
        fragment_dict['tables'] = [t for t in fragment_dict['fragments'] if (t not in fragment_dict['attributes']) & (t not in fragment_dict['functions']) & (t not in fragment_dict['literals'])]

    return fragment_dict

def get_topN(ls, N=None):
    '''
    Get top-N elements from a list
    :param ls (list): a list of tokens (i.e., string), in our case
    :param N (int):

    :return (list): a list of tokens
    '''
    # Return the whole list if N is none
    if N is None:
        return ls
    
    if N > len(ls):
        N = len(ls)
    return ls[:N]

def sort_fragments(token_dict, lookup_dict):
    '''
    Generate top-N predictions for a given query
    We need N as N is independent of K

    :param token_dict (dict): a dictionary, (k:token, v:score)
    :param N (int):
    
    :return sorted_dict (dict): a dict of fragments, (k:fragment type, v:list)
    '''
    tokens = list(token_dict.keys())
    fragment_dict = get_fragments(tokens, lookup_dict)
    
    sorted_dict = {}
    fragment_types = list(fragment_dict.keys())
    for f_type in fragment_types:
        f_dict = {k: v for k, v in token_dict.items() if k in fragment_dict[f_type]}
        sorted_f_dict = list(sort_dict(f_dict, reverse=True).keys())
        #sorted_dict[f_type] = get_topN(sorted_f_dict, N)
        sorted_dict[f_type] = sorted_f_dict
    
    return sorted_dict

def get_fragment_dict(voc, trgs, preds, if_beam_search, lookup_dict=None):
    '''
    Get fragment dict from K beams
    :param targets (list): a list of target queries. Each query is a list of tokens
    :param preds_list (list): a list of predicted queries. Each query is a dict (k:token, v:score)

    :return trg_dicts (list): a list of fragment dict
    :return pred_dicts (list): a list of fragment dict
    '''
    trg_dicts = []
    pred_dicts = []
    for i in range(len(trgs)):
        trg_dict = get_fragments(trgs[i], lookup_dict)

        if if_beam_search:
            pred_list = get_token_dict(preds[i], voc)
            pred_dict = sort_fragments(pred_list, lookup_dict)
        else:
            pred_dict = get_fragments(preds[i], lookup_dict)
            
        trg_dicts.append(trg_dict)
        pred_dicts.append(pred_dict)
    return trg_dicts, pred_dicts

def get_token_dict(results, voc):
    '''
    Generate a token dict from BS results
    :param results (list): a list of dictionary, which is the output of generate()
    :return token_dict (dict): dictionary with token and positional score
    '''
    token_dict = {}

    # For each beam
    for result in results:
        # Get tokens, i.e., indecies 
        tokens = result['tokens'].detach().cpu().numpy()
    
        # Mask the duplicates in the same beam
        dup_mask = np.zeros_like(tokens, dtype=bool)
        dup_mask[np.unique(tokens, return_index=True)[1]] = True
    
        # Get log prob, turn into prob for summation
        positional_scores = result['positional_scores'].detach().cpu().numpy()
        if (positional_scores <= 0).all():
            # If log prob applied 
            positional_scores = np.exp(positional_scores)
    
        for i in range(len(tokens)):
            token = tokens[i]
            word = voc.index2word[token]
            # Add new word
            if word not in token_dict:
                token_dict[word] = positional_scores[i]
            # Only consider the first occurance of the word per beam
            elif dup_mask[i]: token_dict[word] += positional_scores[i]
    
    return token_dict

def precision(trg, pred):
    '''
    Calculate precision at each k.
    :param trg (1d list): a list of tokens
    :param pred (2d list): a (K-length) list of lists of tokens
    :return: precision at k
    '''
    trg = set(trg)
    pred = set(pred)
    TP = len(list(trg & pred))
    
    # Return precision
    pred_len = len(pred)

    if len(trg) > 0:
        if pred_len > 0:
            return TP/pred_len
        # When no preds but there are trgs
        else: return 0
    # Ignore when there are no trgs
    return np.nan

def recall(trg, pred):
    '''
    Calculate recall at each k. 
    :param trg (1d list): a list of tokens (i.e., one of fragments, tab, att, fun, and lit). 
    :param pred (2d list): a (K-length) list of lists of tokens, same as the trg. 
    :return: recall at k.
    '''
    trg = set(trg)
    pred = set(pred)
    TP = len(list(trg & pred))
    
    # Return recall
    if len(trg) > 0:
        return TP/len(trg)
    return np.nan

def f_measure(prec, recall):
    '''
    Return f-measure from precision and recall
    :param prec (2d numpy array): [i,k]
    :param recall (2d numpy array): [i,k]
    
    :return (2d numpy array): [i,k] 

    Use:
        prec = calc_fragment_metric(trg, [pred, pred], max_k, precision_at_k)[0] #[0] is to get fra_list only
        recall = calc_fragment_metric(trg, [pred, pred], max_k, recall_at_k)[0]
        f = f_measure(prec, recall)
        f[:,0].mean() # 0 is to get k=1 only
    '''
    a = (2*prec*recall)
    b = (prec+recall)
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def calc_fragment_metric(trg, pred, N, metric_func):
    '''
    Calculate a given metric, either precision or recall.
    :param trg: a list of lists of tokens, i.e., 2d array
    :param pred: a list of lists (i.e., top n) of lists of tokens, i.e., 3d array. Same length as trg
    :param N (None or int): calculate for the whole pred set if None
    :param metric_func (function): the given metric, either precision_at_k or recall_at_k 
    
    :return dict_metric (dict): precision of fragment prediction defined by the formula
    '''
    if N is not None:
        topN = N
    else: topN = 1

    # Initiate two sets of 2d arrays, one set for intermediate results
    dict_N = {}
    for t in F_TYPES:
        dict_N[t] = np.empty((len(trg),topN,))
        dict_N[t].fill(np.nan)
    
    '''
    dict_metric = {}
    for t in F_TYPES:
        dict_metric[t] = np.empty((len(trg),topN,))
        dict_metric[t].fill(np.nan)
    '''

    # Calculate precision at each n. Note that this is an intermediate step
    for i in range(len(trg)):
        trg_dict = trg[i]
        pred_dict = pred[i]

        for n in range(topN):
            for t in F_TYPES:
                trg_ls = trg_dict[t]
                pred_ls = pred_dict[t]

                # Only consider targets with sufficient fragments 
                #if len(trg_ls) <= n:
                #    trg_ls = []

                if N is None:
                    # Evaluate the whole set
                    dict_N[t][i,n] = metric_func(trg_ls, pred_ls)
                else:
                    dict_N[t][i,n] = metric_func(trg_ls, get_topN(pred_ls, n+1))
    
    '''
    # Once precision at n is populated, compute precision
    for i in range(len(trg)):
        for n in range(topN):
            for t in F_TYPES:
                dict_metric[t][i,n] = (sum(dict_N[t][i, :n+1]) / (n+1))
    '''

    return dict_N #dict_metric

def write_eval(fp, model_name, dataset_name, data_size, max_k, time_in_sec, trg, pred):
    '''
    :param fp: string, filepath
    :param model_name: string, model name
    :param dataset_name: string, dataset name
    :param data_size: string, number of pairs
    :param k: int, k
    :param time_in_sec: string, runtime in second
    :param trg: a list of lists of tokens, i.e., 2d array
    :param pred: a list of lists (i.e., top k) of lists of tokens, i.e., 3d array. Same length as trg
    '''
    with open(fp, 'a', newline='') as cvfile:
        fieldnames = ['model_name', 'dataset_name', 'data_size', 'k', 'time_in_sec', 'recall_fra', 'recall_tab', 'recall_att', 'recall_fun', 'recall_lit']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        recall = np.asarray(calc_recall(trgs, preds, max_k=max_k))
        prec = np.asarray(calc_precision(trgs, preds, max_k=max_k))
        recall = recall.T
        prec = prec.T
        for k in range(max_k):
            r = list(recall[k])
            writer.writerow({'model_name': model_name, 'dataset_name': dataset_name, 'data_size': data_size, 'k': str(k), 'time_in_sec': str(time_in_sec), 'recall_fra': r[0], 'recall_tab': r[1], 'recall_att': r[2], 'recall_fun': r[3], 'recall_lit': r[4]})
    print("Successfully logged results to {}".format(fp))

def mean_confidence_interval(data, confidence=0.95):
    '''
    Return confidence interval
    :param data (2d list): [i,k], e.g., f-measure data

    :return m (1d array): mean value of data
    :return h (1d array): interval value

    Use:
        datafile = DIR_PATH+folder+file
        df = pd.read_csv(datafile, low_memory=False)
        a = np.asarray(mean_confidence_interval(df.values))
        a = a.T
        df = pd.DataFrame(a[:10], columns=['m', 'h'])
        df['k'] = [(i+1) for i in range(max_k)]
    '''
    a = 1.0 * np.array(data)

    n = len(a)
    m, se = np.nanmean(a, axis=0), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h
