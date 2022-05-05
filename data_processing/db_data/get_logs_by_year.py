import datetime 
import csv

from utils import get_log_with_IDs, get_log_by_time, get_log_with_IDs_and_time, find_occurrences

READ_FROM = 'F:\data\sdssweblogs'
WRITE_TO = 'F:\data\processed\year'

def get_logs_by_year(year, verbose=True):
    '''
    Segment the year data from the original data first
    This step is the first step of pipeline.py to reduce the size of the files 
    that we need to iterate through
    :param year: an int indicates the year
    '''
    # Make start and end period string
    start = '1/1/' + str(year)
    end = '1/1/' + str(year+1)
    '''
    # Get sessions with time frame
    filepath = READ_FROM + '\session.csv'
    newfile = WRITE_TO + '\session.csv'
    get_log_by_time(filepath, newfile, 3, '%m/%d/%Y', start, end)

    # Get sessionIDs
    fp = WRITE_TO + '\session.csv'
    with open(fp, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        sessionIDs = [int(row[0]) for row in reader]
    sessionIDs = set(sessionIDs) # Remove duplicates
    if verbose:
        print('There are {} sessions in {}'.format(len(list(sessionIDs)), year))

    # Get sessionlog with sessionIDs
    filepath = READ_FROM + '\sessionlog.csv'
    newfile = WRITE_TO + '\sessionlog.csv'
    get_log_with_IDs(filepath, newfile, sessionIDs, 0)
    '''

    fp = WRITE_TO + '\sessionlog.csv'
    hitIDs = []
    sqlIDs = []
    with open(fp, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[4]) == 0: # hitID if sessionlog.type == 0
                hitIDs.append(int(row[5]))
            else:
                sqlIDs.append(int(row[5]))
    hitIDs = set(hitIDs) # Remove duplicates
    sqlIDs = set(sqlIDs) # Remove duplicates
    if verbose:
        print('There are {} hitIDs in {}'.format(len(list(hitIDs)), year))
        print('There are {} sqlIDs in {}'.format(len(list(sqlIDs)), year))

    # Get weblog with hitIDs
    filepath = READ_FROM + '\weblog.csv'
    newfile = WRITE_TO + '\weblog.csv'
    get_log_with_IDs_and_time(filepath, newfile, hitIDs, 0, 7, '%Y-%m-%d', start, end)

    # Get sqllog with sqlIDs
    filepath = READ_FROM + '\sqllog.csv'
    newfile = WRITE_TO + '\sqllog.csv'
    get_log_with_IDs_and_time(filepath, newfile, sqlIDs, 0, 7, '%Y-%m-%d', start, end)

    if verbose:
        print('Successfully get logs for year {}'.format(year))

