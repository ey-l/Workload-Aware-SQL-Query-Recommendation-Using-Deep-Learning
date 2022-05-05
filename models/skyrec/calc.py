from options import TABLES, VIEWS

import re

# for crude summary and skyrec summary
mergedList = TABLES + VIEWS
mergedList = [t.lower() for t in mergedList]

def crudeCalc(query, sqlID, writer):
    for i in mergedList:

        if re.search(r"\b" + i + r"\b", query):
            #print(query + " has:")
            #print(i)
            row = [str(sqlID), str(query), str(i), 1]

            writer.writerow(row)


def detailedCalc(query, sqlID, writer, tuple):
    #tupleResult = tuple.decode("utf-8")
    row = [str(sqlID), str(query), str(tuple), 1]
    writer.writerow(row)


def skyRecCalc(query, sqlID, writer, tupleCount):
    for i in mergedList:

        if re.search(r"\b" + i + r"\b", query):
            #print(query + " has:")
            #print(i)
            row = [str(sqlID), str(query), str(i), str(tupleCount)]

            writer.writerow(row)

