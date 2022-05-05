import pandas as pd
import numpy as np
import datetime 
import csv
from utils import get_log_with_IDs
import sys

maxInt = sys.maxsize

'''
Decrease the maxInt value by factor 10 as long as the OverflowError occurs.
'''
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def get_sqlstatement(WRITE_TO):
    '''
    Get sql statements with statementIDs from sqllog records
    Encounter field larger than field limit (131072) error when executing in the notebook
    '''
    READ_FROM = 'F:\data\processed' #'F:\data\sdssweblogs'
    #sqllog = pd.read_csv(WRITE_TO + '\sqllog.csv', header=0, low_memory=False)
    sqllog = pd.read_csv(WRITE_TO + '\\humansession.csv', header=0, low_memory=False)
    statementIDs = set(list(sqllog.statementid.values))

    # Get the corresponding sqlstatement records
    filepath = READ_FROM + '\sqlstatement.csv'
    newfile = WRITE_TO + '\sqlstatement.csv'
    get_log_with_IDs(filepath, newfile, statementIDs, 0)

if __name__ == "__main__":
    #WRITE_TO = 'F:\data\processed\sampled'
    WRITE_TO = 'F:\data\processed\\test'
    get_sqlstatement(WRITE_TO)
