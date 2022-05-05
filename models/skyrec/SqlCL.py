#!python3
""">> sqlcl << command line query tool by Tamas Budavari <budavari@jhu.edu>
Usage: sqlcl [options] sqlfile(s)

Options:
        -s url	   : URL with the ASP interface (default: pha)
        -f fmt     : set output format (html,xml,csv - default: csv)
        -q query   : specify query on the command line
        -l         : skip first line of output with column names
        -v	   : verbose mode dumps settings in header
        -h	   : show this message"""

import pandas as pd
import calc
import csv

from timeout import timeout
from options import TIMEOUTLIST

formats = ['csv','xml','html']

astro_url='http://skyserver.sdss3.org/public/en/tools/search/x_sql.aspx'
public_url='http://skyserver.sdss3.org/public/en/tools/search/x_sql.aspx'

default_url=public_url
default_fmt='csv'

def usage(status, msg=''):
    "Error message and usage"
    if msg:
        print('-- ERROR: %s' % msg)
    sys.exit(status)

def filtercomment(sql):
    "Get rid of comments starting with --"
    import os
    fsql = ''
    for line in sql.split('\n'):
        fsql += line.split('--')[0] + ' ' + os.linesep
    return fsql

#@timeout(10) use only when data loading
def query(sql,url=default_url,fmt=default_fmt):
    "Run query and return file object"
    import urllib.parse
    import urllib.request
    fsql = filtercomment(sql)
    params = urllib.parse.urlencode({'cmd': fsql, 'format': fmt})
    return urllib.request.urlopen(url+'?%s' % params)    

def write_header(ofp,pre,url,qry):
    import  time
    ofp.write('%s SOURCE: %s\n' % (pre,url))
    ofp.write('%s TIME: %s\n' % (pre,time.asctime()))    
    ofp.write('%s QUERY:\n' % pre)
    for l in qry.split('\n'):
        ofp.write('%s   %s\n' % (pre,l))
    
def main(argv):
    "Parse command line and do it..."
    import os, getopt, string
    
    queries = []
    queryIDs = []
    url = os.getenv("SQLCLURL",default_url)
    fmt = default_fmt
    writefirst = 1     # whether keep first line of output with column names
    verbose = 0
    calcFormat = 'crude' # calculation format
    
    # Parse command line
    try:
        optlist, args = getopt.getopt(argv[1:],"s:f:q:b:c:l:v:")
    except getopt.error as e:
        usage(1,e)
        
    for o,a in optlist:
        if   o=='-s': url = a
        elif o=='-f': fmt = a
        elif o=='-q': queries.append(a)
        elif o=='-b':
            # -b specifies file name
            test = pd.read_csv(a)

            # remove column name "statement"
            file_queries = test['statement'].head(1000)
            for i in file_queries:
                queries.append(i)

            file_queryIDs = test['sqlID'].head(1000)
            for i in file_queryIDs:
                queryIDs.append(i)

        elif o =='-c':
            calcFormat = a
        elif o=='-l': writefirst = 0
        elif o=='-v': verbose += 1
        else: usage(0)
        
    if fmt not in formats:
        usage(1,'Wrong format!')

    '''
    for fname in args:
        try:
            queries.append(open(fname).read())
        except IOError as e:
            usage(1,e)
    '''


    with open ('data/'+calcFormat+'_calc.csv', "a") as resultFile:
        writer = csv.writer(resultFile, dialect='excel')

        # Run all queries sequentially
        for count, qry in enumerate(queries):

            if verbose:
                write_header(sys.stdout,'#',url,qry)

            if calcFormat == 'crude' and str(qry).upper().startswith('SELECT'):
                sqlID = queryIDs[count]

                calc.crudeCalc(qry, sqlID, writer)

            if calcFormat == 'detailed' and str(qry).upper().startswith('SELECT') and not str(qry) in TIMEOUTLIST:
                sqlID = queryIDs[count]

                print(str(qry)+'\n')

                try:
                    file = query(qry, url, fmt)
                    # read table
                    file.readline()
                    # read column names
                    line = file.readline()

                    if not line.startswith(bytes("<!DOCTYPE", 'utf-8')):
                        line = file.readline()
                        tupleNumber = 0
                        # max-size limit on tupleNumber
                        while tupleNumber < 1000 and line:
                            line = file.readline()
                            tupleNumber += 1
                            calc.detailedCalc(qry, sqlID, writer, line)
                except Exception:
                    print("timeout occurred for "+ qry)

            if calcFormat == 'skyrec' and str(qry).upper().startswith('SELECT') and not str(qry) in TIMEOUTLIST:
                sqlID = queryIDs[count]

                print(str(qry) + '\n')

                try:
                    file = query(qry, url, fmt)
                    # read table
                    file.readline()
                    # read column names
                    line = file.readline()

                    if not line.startswith(bytes("<!DOCTYPE", 'utf-8')):
                        line = file.readline()
                        tupleNumber = 0
                        while line:
                            line = file.readline()
                            tupleNumber += 1

                        calc.skyRecCalc(qry, sqlID, writer, tupleNumber)
                except Exception:
                    print("timeout occurred for " + qry)


if __name__=='__main__':
    import sys
    main(sys.argv)





