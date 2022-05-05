import os
import sys
sys.path.append('../../models/')
from imports import *
from skyrec import options

import sqlparse
from sqlparse.tokens import Name, Number, DML, Keyword
from sqlparse.sql import IdentifierList, Identifier
from moz_sql_parser import parse
from moz_sql_parser import format as fo
import json
from collections import abc
import ast
import numbers

def templatify(json, t = 0, context = "s"):
# """Takes a query and replaces variables/literals with ATT, TAB, NUM, STR
# Args:
# json is the parsed query as a dict 
# t is the type of passed file: 0 is dict, 1 is list
# """
    arith = ["eq", "gt", "gte", "lt", "lte", "between", 'str', 'floor', 'neq', 'gte', 'lte', 'missing', 'exists'] #'div', 'mul', 'sub', 'add' 
    math = ['div', 'mul', 'sub', 'add']
    functions = ["min", "max", "avg", "count", "sum", "distinct", "join",'datetime','dateadd','convert','month','day','year','from', 'abs','len','stdev','log','isdate']
    if t == 0:
        for k, v in json.items():
            if (k.lower() in [fun.lower() for fun in options.FUNCTIONS]) or ('dbo.' in k.lower()):
                json[k] = "ATT"
                json["FUN"] = json.pop(k)
            if (k == "value" or k in functions or k in arith) and not isinstance(v, dict):
                if context == "s":
                    json[k] = "ATT"
                if (k in arith) and isinstance(v, list):
                    for l in range(len(v)):
                        if isinstance(v[l], str):
                            if v[l] != "ID" and v[l] != "NUM":
                                v[l] = "ATT"
                elif (k in math) and isinstance(v, list):
                    for l in range(len(v)):
                        if isinstance(v[l], str):
                            if v[l] != "ID" and v[l] != "NUM":
                                v[l] = ["ATT", "ATT"]
                elif context == "f":
                    json[k] = "TAB"
                elif context == "g" or context == "o" or context == "li":
                    json[k] = "ATT"
            if k == "literal":
                json[k] = "STR"
            if k == "name":
                json[k] = "ALIAS"
            if k in ["join", 'from', 'left join', 'right join']:
                if not isinstance(v, dict) and not isinstance(v, list):
                    json[k] = "TAB"
                context = "f"
            if k == "select":
                if not isinstance(v, dict) and not isinstance(v, list):
                    json[k] = "ATT"
                context = "s"
            if k in math+arith:
                json[k] = ["ATT", "NUM"]
            if k == "groupby":
                context = "g"
            if k == "having":
                context = "h"
            if k == "orderby":
                context = "o"
            if k == "like" or k == "in":
                context = "li"
                json[k][0] = "ATT"
                json[k][1] = "STR"
            if isinstance(v, dict):
                templatify(v, 0, context)
            elif isinstance(v, list):
                templatify(v, 1, context)
    else:
        for i in range(len(json)):
            item = json[i]
            if isinstance(item, dict):
                templatify(item, 0, context)
            elif isinstance(item, list):
                templatify(item, 1, context)
            else:
                json[i] = 'ATT'

def templatify_wrapper(q):
    try:
        if q != "ERROR":
            q = ast.literal_eval(q) # str to dict
            q = sort_tree(q) # sort before templify
            templatify(q)
            q = fo(q)
        return q
    except:
        #print('Error')
        return 'TEMPLATE ERROR'

def sort_mixed_list(ls):
    '''
    Sort a list of mixed-type elements
    Example:
        ls = ['ATT', 'NUM', 2, {'value': 'ATT'}, {'fun': 'LIT'}]
    '''
    return sorted(ls, key=lambda x: (x is not None, '' if isinstance(x, numbers.Number) else (str(x) if isinstance(x, dict) else type(x).__name__), x))

def sort_tree(tree):
    '''
    Sort an AST. moz_sql_parser.parse() only turns query into a dict. 
    Without "sorting", it's difficult to compare ASTs.

    :param tree (dict): AST in a dictionary, which is the output of moz_sql_parser.parse()

    Note:
        Possible value types: str, dict, dict-only list, mixed-type list
    '''
    if isinstance(tree, str):
        return tree
    if isinstance(tree, numbers.Number):
        return 'NUM'
    if isinstance(tree, list):
        # list of dicts only
        if all(isinstance(x, dict) for x in tree):
            return sorted(tree, key=lambda d: list(d.keys()))
        # list of str-dict mix 
        else: 
            return sort_mixed_list(tree)
    
    for node in tree:
        tree[node] = sort_tree(tree[node])
        
    return dict(sorted(tree.items()))

def process_template(t):
    t = re.sub(r"([=><,!?()+-/])", r" \1 ", t)
    t = t.split(" ")
    t = " ".join(t)
    t = re.sub(r"\s+", r" ", t).strip()
    t = re.sub('(ATT )+', 'ATT ', t)
    t = re.sub('(ATT , )+', 'ATT , ', t)
    t = re.sub('(AS ALIAS )', ' ', t)
    t = re.sub('(NUM )+', 'NUM ', t)
    t = re.sub(r"\s+", r" ", t).strip()
    return t

def format_q(q):
    try:
        q = ast.literal_eval(q)
        return fo(q)
    except:
        return "ERROR"