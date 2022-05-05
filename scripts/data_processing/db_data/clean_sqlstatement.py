import numpy as np
import datetime 
import csv
from utils import find_occurrences

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

def main():
    '''
    Used to clean up the raw sqlstatement.csv downloaded from the archive repo
    The raw csv file is messed up since there is no double quotations around the 
    sql statement field
    Since the sql statements generally contain commas, the csv couldn't load properly
    '''
    fp = 'F:\data\sdssweblogs\sqlstatement-raw.csv'
    newfile = 'F:\data\processed\sqlstatement.csv'
    count = 0
    with open(fp, 'r', errors='ignore') as f, open(newfile, 'w', newline='') as newfile:
        reader = csv.reader(f)
        writer = csv.writer(newfile)
        new_row = []
        prev_row = []
        for row in reader:
            if count == 0:
                new_row = row
                count += 1
                next(reader)
            elif len(row) > 0:
                if row[0].isdigit():
                    if int(row[0]) >= count and int(row[0]) <= count + 10000: # Check if it's in a range
                        if len(prev_row) > 0 or new_row[0].isdigit():
                            sql_statement = new_row[1]
                            occurs = find_occurrences(sql_statement, ',')
                            if len(occurs) > 2:
                                i = occurs[-3]
                                new_row[1] = sql_statement[:i]
                                new_row += sql_statement[i+1:len(sql_statement)].split(',')
                        writer.writerow(new_row) # Write new_row
                        if new_row[0].isdigit():
                            count = int(new_row[0]) # Update count
                        prev_row = []
                        new_row = [row[0], ','.join(row[1:])]
                        #count += 1
                    else:
                        sql_index = len(new_row)-1
                        prev_row = row # Save as previous row
                        sql_statement = ' ' + ','.join(row) + ' \n '
                        new_row[sql_index] += sql_statement
                else:
                    sql_index = len(new_row)-1
                    prev_row = row # Save as previous row
                    sql_statement = ' ' + ','.join(row) + ' \n '
                    new_row[sql_index] += sql_statement
            if count%500000 == 0:
                print('Processed {} rows in total'.format(count))
        print('Successfully processed and saved data to {}'.format(newfile))

if __name__ == "__main__":
    main()
    