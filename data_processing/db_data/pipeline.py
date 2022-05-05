import os
import sys

'''
This is a toy pipeline for getting sqllog with statements
We take the following steps:
    1. Get sessions that occurred in January 2009 from session
    2. Based on session.sessionID, get session logs from sessionlog
    3. Based on sessionlog.ID, get the corresponding weblog records and sqllog records. If sessionlog.type is 0, sessionlog.ID points to a weblog record; if sessionlog.type is 1, sessionlog.ID points to a sqllog record
        Identify the session class (i.e., one of BOT, ADMIN, BROWSER, ADMIN, ANONYMOUS, UNKNOWN) using weblog and webagentstring.
        We do a majority vote here, following Zainab’s steps
    4. The point of getting the session class type is to identify if a session is human-generated
    5. Based on sqllog.statementID, get statements from sqlstatement
    6. Join sqllog with sqlstatement on statementID
    7. Join sqllog with sessionlog on ID to map statements to their session
    8. Some data processing (e.g., removing redundant/unused columns) before saving
'''

from utils import get_log_with_IDs, get_log_by_time, get_log_with_IDs_and_time, find_occurrences
from get_logs_by_year import get_logs_by_year
from get_sqlstatement import get_sqlstatement
import pandas as pd
import csv
import numpy as np
import datetime
import time

READ_FROM = 'F:\data\processed\year'
WRITE_TO = 'F:\data\processed\\test'
SESSION_CLASS = ['sessionID', 'class']

'''
For parsing theTime attribute
'''
def parse_theTime(t):
    occurs = find_occurrences(t, '.')
    i = occurs[-1]
    new_time = datetime.datetime.strptime(t[:i], '%Y-%m-%d  %H:%M:%S')
    return new_time

def main():
    # Segment the year data from the original data first
    year = 2010
    #get_logs_by_year(year)

    # For each month in 2010
    for month in range(1, 13):
        # Timer
        timer_start = time.time()
        # Make start and end period string
        start = str(month) + '/1/' + str(year)
        end = str(month+1) + '/1/' + str(year)
        if month == 12:
            end = '1/1/' + str(year+1)
        
        # Get sessions with time frame
        filepath = READ_FROM + '\session.csv'
        newfile = WRITE_TO + '\session.csv'
        get_log_by_time(filepath, newfile, 3, '%m/%d/%Y', start, end)

        # Get sessionIDs
        session = pd.read_csv(newfile, header=0, low_memory=False)
        sessionIDs = session['sessionID'].values
        sessionIDs = set(sessionIDs)

        # Get sessionlog with sessionIDs
        filepath = READ_FROM + '\sessionlog.csv'
        newfile = WRITE_TO + '\sessionlog.csv'
        get_log_with_IDs(filepath, newfile, sessionIDs, 0)

        '''
        Get the hitIDs from the full sessionlog
        If sessionlog.type = 0, then sessionlog.ID points to weblog.hitID
        '''
        sessionlog0 = pd.read_csv(WRITE_TO + '\sessionlog.csv', header=0, low_memory=False)
        sessionlog0 = sessionlog0[sessionlog0.type == 0]
        hitIDs = set(sessionlog0.ID.values)

        # Get the corresponding weblog records and make sure sqllog records retrived by sqlIDs are within Jan. 2009
        filepath = READ_FROM + '\weblog.csv'
        newfile = WRITE_TO + '\weblog.csv'
        get_log_with_IDs_and_time(filepath, newfile, hitIDs, 0, 7, '%Y-%m-%d', start, end)

        # Load weblog records
        weblog = pd.read_csv(WRITE_TO + '\weblog.csv', low_memory=False, header=0)

        # Load merged webagentstring
        agentstr = pd.read_csv(WRITE_TO + '\webagentstring.csv', header=0)

        # Identify weblog records' class with agentstringID 
        weblog = pd.merge(weblog, agentstr, on='agentStringID')

        # Identify session class based on weblog records
        weblog = weblog[['hitID', 'class']]
        
        # Get the hitIDs from the full sessionlog
        sessionlog0 = pd.merge(sessionlog0, weblog, left_on='ID', right_on='hitID', how='left')

        # Keep the sessionID and class
        sessionlog0 = sessionlog0[SESSION_CLASS]

        # Take the majority vote to determine the session class
        class_labels = sessionlog0.groupby(['sessionID','class']).size().reset_index(name='counts')
        class_labels = class_labels.groupby('sessionID').apply(lambda g: g[g['counts'] == g['counts'].max()]['class'])
        class_labels = class_labels.reset_index()[SESSION_CLASS]
        class_labels = class_labels.drop_duplicates(subset='sessionID', keep='first') # a dataframe

        # Find bot sessions when there is even one bot weblog record in that session
        bot_labels = sessionlog0.groupby(['sessionID']).apply(lambda g: g['class'].eq('BOT').any()).reset_index(name='ifBOT')

        # Merge and set session class to BOT when there is even one bot weblog record in that session
        class_labels = pd.merge(class_labels, bot_labels)
        class_labels['class'] = np.where(class_labels.ifBOT == True, 'BOT', class_labels['class'])
        class_labels = class_labels[SESSION_CLASS]

        # Merge the class labels with the sampled sessionlog
        sessionlog = pd.read_csv(WRITE_TO + '\sessionlog.csv', low_memory=False, header=0)
        sessionlog = pd.merge(sessionlog, class_labels, on='sessionID', how='left')

        print("There are {} sessions out of {} sessions that don't have a single webhit".format(len(list(set(list(sessionlog[sessionlog['class'].isnull()].sessionID.values)))), len(list(set(list(sessionlog.sessionID.values))))))

        # Label sessions with no webhits
        sessionlog['class'] = np.where(sessionlog['class'].isnull(), 'NO_WEBHITS', sessionlog['class'])

        # Get sqlIDs with identified class
        sessionlog1 = sessionlog[sessionlog.type == 1]
        sessionlog1 = sessionlog1[['sessionID', 'rankinsession', 'class', 'ID']]

        # Get sqlIDs
        sqlIDs = set(list(sessionlog1.ID.values))

        # Make sure sqllog records retrived by sqlIDs are within the 2009 range
        filepath = READ_FROM + '\sqllog.csv'
        newfile = WRITE_TO + '\sqllog.csv'
        get_log_with_IDs_and_time(filepath, newfile, sqlIDs, 0, 7, '%Y-%m-%d', start, end)

        # Get SQL statement from sqlIDs
        get_sqlstatement(WRITE_TO)

        # Load sqllog and sqlstatement and keep useful columns
        sqllog = pd.read_csv(WRITE_TO + '\sqllog.csv', header=0, low_memory=False)
        sqllog = sqllog[['sqlID', 'theTime', 'dbname', 'statementID', 'error']]
        sqlstatement = pd.read_csv(WRITE_TO + '\sqlstatement.csv', header=0, low_memory=False, encoding='ISO-8859-1')
        sqlstatement = sqlstatement[['statementID', 'statement', 'TemplateID']]

        # Join sqllog to get statement and session info
        sqllog = pd.merge(sqllog, sqlstatement, on='statementID', how='left')
        sqllog = pd.merge(sqllog, sessionlog1, left_on='sqlID', right_on='ID', how='left')

        # Data processing before writing to folder
        sqllog['theTime'] = sqllog['theTime'].apply(parse_theTime)
        del sqllog['ID'] # Remove the join key
        sqllog = sqllog[sqllog.error == 0] # Remove query logs with errors
        sqllog.loc[:,'dbname'] = sqllog.dbname.str.lower() # dbname to lowercases
        sqllog = sqllog[sqllog.dbname.str.contains('^bestdr\d$')] # Remove query logs that are not operated on an SDSS database
        sqllog = sqllog.groupby('sessionID').filter(lambda x: len(x) > 1) # Remove queries in sessions that are less than one error-free query long

        # Save to folder WRITE_TO
        if month == 1:
            sqllog.to_csv(WRITE_TO + '\sessionsql.csv', index=None)
        else:
            sqllog.to_csv(WRITE_TO + '\sessionsql.csv', index=None, mode='a', header=False) # Continue writing

        # Timer end
        timer_end = time.time()
        print('Month {} took {} mins'.format(month, (timer_end - timer_start) / 60))

    print('Successfully saved sessionsql.csv to {}'.format(WRITE_TO))

if __name__ == "__main__":
    main()

