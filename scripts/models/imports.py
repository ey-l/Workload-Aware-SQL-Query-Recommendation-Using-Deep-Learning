# Modelling
import numpy as np
import math, copy, time
import torch
from torch import optim
from torch.jit import script, trace
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import wandb
from sklearn.metrics.pairwise import cosine_similarity

# Data wrangling
from io import open
import itertools
from itertools import count
import re
import sqlparse
import pandas as pd
import datetime
import csv
import random
import os
import sys
import unicodedata
import codecs
import scipy
#from scipy import stats
from queue import PriorityQueue
import operator
from collections import Counter, OrderedDict
import pprint

# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
#import seaborn as sns

# Time functions
import timeit
