import os
import sys
sys.path.append('../parsers/')
from template_parser import templatify_wrapper, process_template, format_q
sys.path.append('../../models/')
from imports import *

DIR_PATH = '/home/eugenie/projects/def-rachelpo/eugenie/data/processed/sdss/'

if __name__ == "__main__":

    df = pd.read_csv(DIR_PATH+'qdict_statements.csv', low_memory=False)
    template = df.copy()
    
    start = timeit.default_timer()
    template['processed'] = template['qdict'].apply(lambda x: format_q(x))
    template['template'] = template['qdict'].apply(lambda x: templatify_wrapper(x))
    template['template_v2'] = template['template'].apply(lambda x: process_template(x))
    stop = timeit.default_timer()
    print('Time in second: ', stop - start)
    
    template.to_csv(DIR_PATH+'model_data/full/sdss_tree_template.csv', index=None)
