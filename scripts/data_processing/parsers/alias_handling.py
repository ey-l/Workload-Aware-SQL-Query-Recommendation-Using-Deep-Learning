import os
import sys
sys.path.append('/home/eugenie/projects/def-rachelpo/eugenie/queryteller/scripts/models/')
from utils import *
from imports import *

from moz_sql_parser import parse
import json
from collections import abc
import re
from numbers import Number
from moz_sql_parser import format as format_query

def handle_alias(sql):
    try:
        sql = alias_parsing_preprocessor(sql)
        pql = parse(sql)
        a = list(extract_table_alias(pql))
        mod_query = replace_attributes(pql, a)
        return format_query(mod_query)
    except:
        return "ERROR"

def alias_parsing_preprocessor(s):
    s = str(s).lower()
    # Replace ID's (8-length BIGINT hex) with a id_token: https://skyserver.sdss.org/dr12/en/help/browser/browser.aspx#&&history=description+PhotoObjAll+U
    s = re.sub(r'0x[0-9a-f]+', r'ID', s)
    # Replace top
    s = re.sub(r"\btop \d+\b", r"", s)
    # Handle into
    s = re.sub(r"\binto ([a-z0-9_@\.]*)\b", r"", s)
    # Replace '..' with '.', Replace '[]'
    s = s.replace("[", "$").replace("].", "$.").replace("]", "$ ").replace("..",".")
    # Remove whitespace after .
    s = re.sub('\. +', '.', s)
    # Replace #
    s = re.sub(r'#', r'', s)
    #print(s)
    return s

def extract(json, t = 0, context = "s", level = 0):
    """Extract variables (attribute names and table names) from query
    Args:
        json is the parsed query as a dict 
        t is the type of 0 is dict, 1 is list
        context is either s for select or f for from
    """
    arith = ["eq", "gt", "gte", "lt", "lte", "between"]
    functions = ["min", "max", "avg", "count", "sum", "distinct", "join", "literal"]
    if t == 0:
        for k, v in json.items():
            if (k == "value" or k in functions or k in arith) and not isinstance(v, dict):
                if context == "s":
                    yield((v, level))
                if (k in arith) and isinstance(v, list):
                    for l in v:
                        if isinstance(l, str):
                            yield((l, level))
#                 elif context == "f":
#                     f.append(v)
                elif context == "g" or context == "o" or context == "li" :
                    yield((v, level))
            if k == "from" or k == "join":
#                 if not isinstance(v, dict) and not isinstance(v, list):
#                     f.append(v)
                context = "f"
            if k == "select":
                if not isinstance(v, dict) and not isinstance(v, list):
                    yield((v, level))
                context = "s"
                level = level + 1
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
                yield from extract(v, 0, context, level)
            elif isinstance(v, list):
                yield from extract(v, 1, context, level)
    else:
        for i in json:
            if isinstance(i, dict):
                yield from extract(i, 0, context, level)
            elif isinstance(i, list):
                yield from extract(i, 1, context, level)

def extract_table_alias(json, t = 0, context = 'o', level = 0):
    """Takes a query and returns a list of table names with aliases and levels
        Args:
        json is the parsed query as a dict 
        alias_lsit is the list of tables and aliases
        t is the type of passed structure: 0 is dict, 1 is list
        context is either f From or o Other
        level is the level of the token in the tree, determines at what level the table occurs
    """
    if t == 0:
        for k, v in json.items():
            if k == "select":
                level = level + 1
            if (k == "from" or k == "join") and isinstance(v, dict):
                # Single table in the from clause
                context = 'f'
                yield((v['value'], v['name'], level))
            elif k == "from" or k == "join" and not isinstance(v, dict):
                # Multiple tables in a list, call function to handle list
                context = 'f'
                yield from extract_table_alias(v, 1, context, level)
            if k != "from":
                context = "o"
            if k != "from" and isinstance(v, dict):
                yield from extract_table_alias(v, 0, context, level)
            if k != "from" and isinstance(v, list):
                yield from extract_table_alias(v, 1, context, level)
    else:
        for i in json:
            if isinstance(i, dict):
                if context == 'f' and ('value' in i):
                    yield((i['value'], i['name'], level))
                yield from extract_table_alias(i, 0, context, level)
            elif isinstance(i, list):
                yield from extract_table_alias(i, 1, context, level)

def replace_alias(att, tables, l):
    """Takes a single attribute and a list of tables and aliases. Returns the attribute + table name isntead of alias.
    Args:
        att is the attribute
        t is the list of all tables and aliases
        l is the attribute level in the query
    """   
    if "." in att:
        temp = att.split(".")
        for t in tables:
            if isinstance(t[0], dict):
                continue
            else:
                if temp[0] == t[1] and l >= t[2]:
                    temp[0] = t[0]
                    att = ".".join(temp)
        return att
    else:
        return att

def replace_attributes(json, tables, t = 0, context = "s", level = 0):
    """Takes a query and a list of tables and aliases. Returns the query with the new attribute names and without table aliases.
    Args:
        json is the parsed query as a dict 
        tables is the list of tables and aliases
        t is the type of passed structure: 0 is dict, 1 is list
        context is either f From or o Other
        level is the level of the token in the tree, determines at what level the table occurs
    """
    arith = ["eq", "gt", "gte", "lt", "lte", "between", "like"]
    functions = ["min", "max", "avg", "count", "sum", "distinct", "join", "literal"]
    if t == 0:
        for k, v in json.items():
            if (k == "value" or k in functions or k in arith) and not isinstance(v, dict):
                if context == "s":
                    json[k] = replace_alias(v, tables, level)
                if (k in arith) and isinstance(v, list):
                    for i, l in enumerate(v):
                        if isinstance(l, str):
                            v[i] = replace_alias(l, tables, level)
                elif context == "g" or context == "o" or context == "li" or context == "ob":
                    json[k] = replace_alias(v, tables, level)
            
            # This is only for user defined/custom functions
            if (k == "value" and context == "s" and isinstance(v, dict)):
                for key, val in v.items():
                    #print(v[key])
                    if isinstance(val, list):
                        for i, l in enumerate(val):
                            val[i] = replace_alias(l, tables, level)
                    elif isinstance(val, str):
                        v[key] = replace_alias(val, tables, level)
            
            if k == "from" or k == "join":
                if isinstance(v, dict):
                    v.pop('name', None)
                context = "f"
            if k == "select":
                level = level + 1
                if not isinstance(v, dict) and not isinstance(v, list):
                    json[k] = replace_alias(v, tables, level) 
                context = "s"
            if k == "groupby":
                context = "g"
            if k == "having":
                context = "h"
            if k == "orderby":
                context = "ob"
            if k == "like":
                context = "li"
            if k == "on":
                context = "on"
            if k == "and":
                context = "and"
            if isinstance(v, dict):
                if context == 'f':
                    v.pop('name', None)
                replace_attributes(v, tables, 0, context, level)
            elif isinstance(v, list):
                replace_attributes(v, tables, 1, context, level)
    else:
        for i in json:
            if isinstance(i, dict):
                if context == 'f':
                    i.pop('name', None)
                replace_attributes(i, tables, 0, context, level)
            elif isinstance(i, list):
                replace_attributes(i, tables, 1, context, level)
    return json