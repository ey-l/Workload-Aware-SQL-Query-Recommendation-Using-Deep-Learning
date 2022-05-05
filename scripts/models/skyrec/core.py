import numpy as np
from scipy.sparse import csr_matrix as sparse_matrix
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import re

from options import TABLES, VIEWS, TIMEOUTLIST

# for crude summary and skyrec summary
mergedList = TABLES + VIEWS

# for skyrec summary and detailed summary
import SqlCL

# goalA: load matrix
def create_user_item_matrix(ratings, user_key="user", item_key="item"):
    n = len(set(ratings[user_key]))
    d = len(set(ratings[item_key]))

    #print(n)
    #print(d)

    user_mapper = dict(zip(np.unique(ratings[user_key]), list(range(n))))
    item_mapper = dict(zip(np.unique(ratings[item_key]), list(range(d))))

    #print(item_mapper)

    user_inverse_mapper = dict(zip(list(range(n)), np.unique(ratings[user_key])))
    item_inverse_mapper = dict(zip(list(range(d)), np.unique(ratings[item_key])))

    user_ind = [user_mapper[i] for i in ratings[user_key]]
    item_ind = [item_mapper[i] for i in ratings[item_key]]

    X = sparse_matrix((ratings["rating"], (user_ind, item_ind)), shape=(n, d))

    return X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# goalB: recommend queries with highest similarity value
# calcFormat: crude / detailed /skyrec
# k: k-most similar queries
def recommend(inputQuery, dir_path, calcFormat='crude', dataset='sdss', k=1):

    #print(inputQuery)

    #step1 call load matrix
    #dir_path = '../../..//scripts//models//skyrec/data/'
    with open(dir_path+calcFormat+'_calc_'+dataset+'.csv', "rb") as f:
        ratings = pd.read_csv(f, names=("sqlID", "user", "item", "rating"))
    flags = ratings['item'].apply(lambda a: not is_number(a))
    ratings = ratings[flags]
    X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = create_user_item_matrix(ratings)
    fragmentList = list(set(list(ratings['item'].values)))
    n, d = X.shape

    #step2 submit for input query as well
    input_user = [0] * d

    if calcFormat == 'crude':
        for i in fragmentList:
            if re.search(r"\b" + i + r"\b", inputQuery) and i in item_mapper.keys():
                ind = item_mapper[i]
                input_user[ind] = 1

    input_user_vector = np.asarray(input_user).reshape(1,-1)
    
    similarities = cosine_similarity(input_user_vector, X).flatten()
    indexes = similarities.argsort()[-k:][::-1]
    out = []
    for i in indexes:
        out.append(user_inverse_mapper[i].split(" "))
    #print(indexes)
    return out