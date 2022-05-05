import os
import sys
sys.path.append('/home/eugenie/projects/def-rachelpo/eugenie/queryteller/scripts/models/')
from utils import *
from imports import *
sys.path.append('../parsers')
from alias_handling import handle_alias
from tree_parser import parse_query

DIR_PATH = '/home/eugenie/projects/def-rachelpo/eugenie/data/processed/sdss/'

if __name__ == "__main__":

    df = pd.read_csv(DIR_PATH+'humansession_parsed.csv', low_memory=False)

    # Sort queries
    df = df.sort_values(['sessionid','rankinsession'])

    # Get 2009 only
    df = df[(df['thetime'] > '2008-12-31') & (df['thetime'] < '2010-01-01')]
    
    # Remove queries with HTML
    urls = df['statement'].apply(lambda x: 'http://' in str(x))
    df = df[urls == False]

    # Drop consecutive duplicates 
    model_data = df.loc[(df.statement_parsed.shift(-1) != df.statement_parsed) & (df.sessionid.shift(-1) == df.sessionid)]

    # Select and get unique statements
    # To minimize runtime
    statements = model_data[['statementid', 'statement', 'error']]
    statements = statements.drop_duplicates('statementid')

    # Parse queries
    start = timeit.default_timer()
    statements['processed'] = statements['statement'].apply(lambda x: handle_alias(x))
    print(len(statements[statements['processed'] == 'ERROR']))
    stop = timeit.default_timer()
    print('Time in second: ', stop - start)

    # Remove comments after being formatted
    statements['processed'] = statements['processed'].apply(lambda x: sqlparse.format(str(x), strip_comments=True).strip())

    # Get unique statements to get dictionary from
    qdict = statements[['statementid', 'processed']]
    qdict = qdict.drop_duplicates('processed')

    # Get and save the dictionary from unique statements
    start = timeit.default_timer()
    qdict['qdict'] = qdict['processed'].apply(lambda x: parse_query(x))
    qdict.to_csv(DIR_PATH+'qdict_statements_year.csv', index=None)
    stop = timeit.default_timer()
    print('Time in second: ', stop - start)

    statements = pd.merge(statements, qdict[['processed', 'qdict']], on ='processed')
    statements = pd.merge(model_data[['sqlid', 'thetime', 'sessionid', 'localrank', 'rankinsession', 'statementid']], statements, on='statementid')

    # Remove queries with errors
    statements = statements[(statements['qdict'] != 'ERROR') & (statements['error'] == 0)]
    
    # Drop sessions with only one query entry
    #statements = statements.groupby('sessionid').filter(lambda x: len(x) > 1)

    # Sort queries
    statements = statements.sort_values(['sessionid','localrank'])

    # Save 
    statements.to_csv(DIR_PATH+'sdss_model_data_year.csv', index=None)