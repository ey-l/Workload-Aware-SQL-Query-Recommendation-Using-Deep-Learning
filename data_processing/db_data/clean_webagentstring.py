import numpy as np
import datetime 
import csv
from utils import find_occurrences

def main():
    '''
    Used to clean up the raw webagentstring.csv downloaded from the archive repo
    The raw csv file is messed up since there is no double quotations around the 
    agentstring field
    Since the agentstrings sometimes contain commas, the csv couldn't load properly
    '''
    fp = 'F:\data\sdssweblogs\webagentstring.csv'
    newfile = 'F:\data\processed\webagentstring.csv'
    count = 0
    with open(fp, 'r', errors='ignore') as f, open(newfile, 'w', newline='') as newfile:
        reader = csv.reader(f)
        writer = csv.writer(newfile)
        new_row = []
        for row in reader:
            if count == 0:
                new_row = row
                count += 1
                next(reader)
            elif len(row) > 0:
                if row[0].isdigit():
                    new_row = [row[0], ','.join(row[1:])]
                    agent_string = new_row[1]
                    occurs = find_occurrences(agent_string, ',')
                    if len(occurs) > 4:
                        i = occurs[-5]
                        new_row[1] = agent_string[:i]
                        new_row += agent_string[i+1:len(agent_string)].split(',')
                    writer.writerow(new_row) # Write new_row
                    count += 1
            if count%100000 == 0:
                print('Processed {} rows in total'.format(count))
        print('Successfully processed and saved data to {}'.format(newfile))

if __name__ == "__main__":
    main()
    