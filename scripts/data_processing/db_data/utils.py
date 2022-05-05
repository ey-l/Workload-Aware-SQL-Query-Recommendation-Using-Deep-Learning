import pandas as pd
import datetime
import csv
import numpy as np

def find_occurrences(s, ch):
    '''
    Count the occurrences of a char in a given string
    :param s: a string
    :param ch: a char 
    :return: a list of indices that are occurrences of ch in s
    '''
    return [i for i, letter in enumerate(s) if letter == ch]

def get_log_by_time(filepath, newfile, timecol, timeformat, start, end, verbose=True):
    '''
    Skip the first two rows of the csv file, get the 2009 data only based on startTime
    :param filepath: file to read from
    :param newfile: file to write to
    :param timecol: an int, indicates the index of 'startTime'
    :param timeformat: a string for time format of the timecol
    :param start: a string for start time that follows '%m/%d/%Y'
    :param end: a string for end time that follows '%m/%d/%Y'

    Usage:
        filepath = 'F:\data\sdssweblogs\session.csv'
        newfile = 'F:\data\processed\session-2009.csv'
        get_log_by_time(filepath, newfile, 3, '%Y-%m-%d', '1/1/2009', '2/1/2009')
        
    For testing on Mac:
        filepath = '/Users/yujinglai/Dropbox/shrquerylogs/sdssweblogs/session.csv'
        newfile = '/Users/yujinglai/Dropbox/shrquerylogs/sdssweblogs/session-2009.csv'
    '''
    start = datetime.datetime.strptime(start,'%m/%d/%Y')
    end = datetime.datetime.strptime(end,'%m/%d/%Y')
    with open(filepath, 'r') as csvfile, open(newfile, 'w', newline='') as newfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(newfile)
        count = 0
        for row in reader:
            if count == 0:
                writer.writerow(row) # Write the field names
                count += 1
            elif row[0].isdigit(): # Check if ID is digit
                startTime = row[timecol].split(' ')[0]
                startTime = datetime.datetime.strptime(startTime, timeformat)
                if startTime < end and startTime >= start:
                    writer.writerow(row)
                count += 1
                if verbose and count%1000000 == 0:
                    print('Processed {} rows in total'.format(count))
    if verbose:
        print('Successfully processed and saved data to {}'.format(newfile))

def get_log_with_IDs(filepath, newfile, IDs, column, verbose=True):
    '''
    Get the corresponding sessionlog data based on the sessionIDs
    :param filepath: file to read from
    :param newfile: file to write to
    :param fieldnames: a list of field/column names in the read/write files
    :param IDs: a set of sessionIDs, used to get the corresponding sessionlog rows
    :param column: an int of the corresponding column index in the filepath; 0 for sessionID, 5 for queryID

    Usage:
        filepath = 'F:\data\sdssweblogs\sessionlog.csv'
        newfile = 'F:\data\processed\sessionlog-2009.csv'
        get_sessionlog_with_IDs(filepath, newfile, sessionIDs, 0)
        
    For testing on Mac:
        filepath = '/Users/yujinglai/Dropbox/shrquerylogs/sdssweblogs/sessionlog.csv'
        newfile = '/Users/yujinglai/Dropbox/shrquerylogs/sdssweblogs/sessionlog-2009.csv'
    '''
    with open(filepath, 'r') as csvfile, open(newfile, 'w', newline='') as newfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(newfile)
        count = 0
        for row in reader:
            if count == 0:
                writer.writerow(row) # Write the field names
                count += 1
            elif len(row) > 0:
                if row[column].isdigit():
                    if int(row[column]) in IDs:
                        writer.writerow(row)
                    count += 1
                    if verbose and count%50000000 == 0:
                        print('Processed {} rows in total'.format(count))
                else:
                    if verbose:
                        print('Skipped an empty row')
    if verbose:
        print('Successfully processed and saved data to {}'.format(newfile))

def get_log_with_IDs_and_time(filepath, newfile, IDs, idcol, timecol, timeformat, start, end, verbose=True):
    '''
    Get the corresponding sessionlog data based on the sessionIDs
    :param filepath: file to read from
    :param newfile: file to write to
    :param fieldnames: a list of field/column names in the read/write files
    :param IDs: a set of sessionIDs, used to get the corresponding sessionlog rows
    :param idcol: an int of the corresponding column index in the filepath; 0 for sessionID, 5 for queryID
    :param timecol: an int, indicates the index of 'startTime'
    :param timeformat: a string for time format of the timecol
    :param start: a string for start time that follows '%m/%d/%Y'
    :param end: a string for end time that follows '%m/%d/%Y'

    Usage:
        filepath = 'F:\data\sdssweblogs\sessionlog.csv'
        newfile = 'F:\data\processed\sessionlog-2009.csv'
        get_sessionlog_with_IDs(filepath, newfile, sessionIDs, 0, 3, '%Y-%m-%d', '1/1/2009', '2/1/2009')
        
    For testing on Mac:
        filepath = '/Users/yujinglai/Dropbox/shrquerylogs/sdssweblogs/sessionlog.csv'
        newfile = '/Users/yujinglai/Dropbox/shrquerylogs/sdssweblogs/sessionlog-2009.csv'
    '''
    start = datetime.datetime.strptime(start,'%m/%d/%Y')
    end = datetime.datetime.strptime(end,'%m/%d/%Y')
    with open(filepath, 'r') as csvfile, open(newfile, 'w', newline='') as newfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(newfile)
        count = 0
        for row in reader:
            if count == 0:
                writer.writerow(row) # Write the field names
                count += 1
            elif len(row) > 0:
                if row[idcol].isdigit():
                    startTime = row[timecol].split(' ')[0]
                    startTime = datetime.datetime.strptime(startTime, timeformat)
                    if int(row[idcol]) in IDs and startTime < end and startTime >= start:
                        writer.writerow(row)
                    count += 1
                    if verbose and count%50000000 == 0:
                        print('Processed {} rows in total'.format(count))
                else:
                    if verbose:
                        print('Skipped an empty row')
    if verbose:
        print('Successfully processed and saved data to {}'.format(newfile))

def get_log_with_columns(filepath, newfile, columns, verbose=True):
    '''
    Get the corresponding columns from a csv file
    :param filepath: file to read from
    :param newfile: file to write to
    :param columns: a list of column indecies

    Usage:
        filepath = 'F:\data\processed\sliced\weblog-2009.csv'
        newfile = 'F:\data\processed\weblog-2009.csv'
        columns = [0, 7, 8, 9, 13, 14, 21]
        get_log_with_columns(filepath, newfile, columns)
        
    For testing on Mac:
        filepath = '/Users/yujinglai/Dropbox/shrquerylogs/sdssweblogs/sessionlog.csv'
        newfile = '/Users/yujinglai/Dropbox/shrquerylogs/sdssweblogs/sessionlog-2009.csv'
    '''
    with open(filepath, 'r', errors='ignore') as csvfile, open(newfile, 'w', newline='') as newfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(newfile)
        count = 0
        for row in reader:
            if len(row) > 3:
                new_row = list(row[i] for i in columns)
                writer.writerow(new_row)
                count += 1
                if verbose and count%50000000 == 0:
                    print('Processed {} rows in total'.format(count))
            else:
                if verbose:
                    print('Skipped an empty row')
    if verbose:
        print('Successfully processed and saved data to {}'.format(newfile))
