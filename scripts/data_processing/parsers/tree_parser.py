import os
import sys
sys.path.append('../../models/')
from imports import *

import sqlparse
from sqlparse.tokens import Name, Number, DML, Keyword
from sqlparse.sql import IdentifierList, Identifier
from moz_sql_parser import parse
from moz_sql_parser import format as fo
import json
from collections import abc
import re
from numbers import Number

def tree_parsing_preprocessor(s):
    s = str(s)
    # Add space around brackets and commas to separate the vocaublary
    s = re.sub(r"([,()/])", r" \1 ", s)
    # Replace digits with a token
    s = re.sub(r" [+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", r" NUM", s)
    # Remove whitespace after .
    s = re.sub('\. +', '.', s)
    # Add space around brackets and commas to separate the vocaublary
    #s = re.sub(r"([,()/])", r" \1 ", s)
    # Remove excessive space
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def parse_query(q):
    try:
        q = tree_parsing_preprocessor(q)
        return parse(q)
    except:
        return 'ERROR'

def extract(json, s, f, t = 0, context = "s"):
# """Extract variables (attribute names and table names) from query
# Args:
# json is the parsed query as a dict 
# s is the list of select attributes
# f is the list of tables
# t is the type of 0 is dict, 1 is list
# context is either s for select or f for from
# """
    arith = ["eq", "gt", "gte", "lt", "lte", "between"]
    functions = ["min", "max", "avg", "count", "sum", "distinct", "join", "literal"]
    if t == 0:
        for k, v in json.items():
            if (k == "value" or k in functions or k in arith) and not isinstance(v, dict):
                if context == "s":
                    s.append(v)
                if (k in arith) and isinstance(v, list):
                    for l in v:
                        if isinstance(l, str):
                            s.append(l)
                        if isinstance(l, Number):
                            print(l)
                elif context == "f":
                    f.append(v)
                elif context == "g" or context == "o" or context == "li" :
                    s.append(v)
            if k == "from" or k == "join":
                if not isinstance(v, dict) and not isinstance(v, list):
                    f.append(v)
                context = "f"
            if k == "select":
                if not isinstance(v, dict) and not isinstance(v, list):
                    s.append(v)
                context = "s"
            if k == "groupby":
                context = "g"
            if k == "having":
                context = "h"
            if k == "orderby":
                context = "b"
            if k == "like":
                context = "li"
#             if k == "limit":
#                 context = "l"
#             if k == "offset":
#                 context = "o"
            if isinstance(v, dict):
                extract(v, s, f, 0, context)
            elif isinstance(v, list):
                extract(v, s, f, 1, context)
    else:
        for i in json:
            if isinstance(i, dict):
                extract(i, s, f, 0, context)
            elif isinstance(i, list):
                extract(i, s, f, 1, context)

def remove_white_space(s):
    l = []
    inside = False
    for i in s:
        if i == '[':
            inside = True
        if i == ']':
            inside = False
        if inside and i == "-":
            i = "_"
        if inside and i == " ":
            i = "_"
        l.append(i)
    return ''.join(l)

def templatify(json, t = 0, context = "s"):
# """Takes a query and replaces variables/literals with ATT, TAB, NUM, STR
# Args:
# json is the parsed query as a dict 
# t is the type of passed file: 0 is dict, 1 is list
# """
    arith = ["eq", "gt", "gte", "lt", "lte", "between"]
    functions = ["min", "max", "avg", "count", "sum", "distinct", "join"]
    if t == 0:
        for k, v in json.items():
            if (k == "value" or k in functions or k in arith) and not isinstance(v, dict):
                if context == "s":
                    json[k] = "ATT"
                if (k in arith) and isinstance(v, list):
                    for l in range(len(v)):
                        if isinstance(v[l], str):
                            v[l] = "ATT"
                        if isinstance(v[l], Number):
                            v[l] = "NUM"
                elif context == "f":
                    json[k] = "TAB"
                elif context == "g" or context == "o" or context == "li" :
                    json[k] = "ATT"
            if k == "literal":
                json[k] = "STR"
            if k == "name":
                json[k] = "ALIAS"
            if k == "from" or k == "join":
                if not isinstance(v, dict) and not isinstance(v, list):
                    json[k] = "TAB"
                context = "f"
            if k == "select":
                if not isinstance(v, dict) and not isinstance(v, list):
                    json[k] = "ATT"
                context = "s"
            if k == "groupby":
                context = "g"
            if k == "having":
                context = "h"
            if k == "orderby":
                context = "b"
            if k == "like":
                context = "li"
            if isinstance(v, dict):
                templatify(v, 0, context)
            elif isinstance(v, list):
                templatify(v, 1, context)
    else:
        for i in json:
            if isinstance(i, dict):
                templatify(i, 0, context)
            elif isinstance(i, list):
                templatify(i, 1, context)

