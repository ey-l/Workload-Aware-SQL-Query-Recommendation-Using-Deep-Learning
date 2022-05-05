import numpy as np
import datetime 
import csv

def main():
    '''
    Used to get BROWSER sessions from sessionsql.csv, which is the
    final product of pipeline.py
    	1. Get queries belong to a BROWSER session
    '''
    fp = 'F:\data\processed\sessionsql\sessionsql_all_2009.csv'
    newfile = 'F:\data\processed\sessionsql\sessionsql_human_2009.csv'
    count = 0
    with open(fp, 'r', errors='ignore') as f, open(newfile, 'w', newline='') as newfile:
        reader = csv.reader(f)
        writer = csv.writer(newfile)
        for row in reader:
            if count == 0:
                writer.writerow(row)
                print(row)
                count += 1
            elif len(row) > 0:
            	if row[9] == 'BROWSER':
            		writer.writerow(row)
            	count += 1
            if count%1000000 == 0:
                print('Processed {} rows in total'.format(count))
        print('Successfully processed and saved data to {}'.format(newfile))

if __name__ == "__main__":
    main()
