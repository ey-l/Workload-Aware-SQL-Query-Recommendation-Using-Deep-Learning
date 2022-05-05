# About Data
Notes on how we get query pairs and templates from SDSS database and the SQLShare.

## SDSS Data

We start from SDSS database, built from the static SDSS workload directory.

### Prelim
We make our own database with the SDSS datasets. Find scripts in ../scripts/db

From pull_human_sessions.sql, we get humansession.csv

In ../notebooks/data/sdss/data_processing/join_parse_queries.ipynb, we replace ID and numeric values with <ID> and <NUM> respectively and make humansession_parsed.csv

### Dataset Extraction

#### Make qdict_statements.csv and sdss_model_data.csv
First, get a dictionary for each unique query statement

We need the dictionary structure to get template(Q) later

Use ../scripts/data_processing/make_model_data.py, get qdict_statements.csv and sdss_model_data.csv 

See ../notebooks/data/sdss/model_data/parse_unique.ipynb for step-by-step breakdown

#### NQP: Make train.csv, val.csv, test.csv
We need <Qi, Qi+1> to train the seq2seq models

Use ../notebooks/data/sdss/model_data/make_pairs.ipynb to get train/val/test dataset from sdss_model_data.csv 

#### Template prediction: Make train.csv, val.csv, test.csv
With qdict_statements.csv, use ../scripts/data_processing/make_template_data.py to get sdss_template.csv

In ../notebooks/data/sdss/model_data/templatify.ipynb, make QueriesWithTemplate.csv with sdss_template.csv and sdss_model_data.csv

With QueriesWithTemplate.csv and train.csv, val.csv, test.csv for NQP, make train.csv, val.csv, test.csv for template prediction

## SQLShare Data
We start with SQLShare workload data downloaded from their website.

### Dataset Extraction

#### Identify sessions
Run find_sessions.ipynb in ../notebooks/data/sqlshare/

The SQLShare dataset doesn't have session IDs. We use the same session definition as SDSS.

#### Process query statements
Run process_to_parse.ipynb in the same dir as above. The SQLShare queries are stored in a format defined by the service creators, which contains the account IDs and other chars that are illegal to the AST parser of our choice. 

Hence we need to process the queries to make it parsable. 

#### Handle alias
Run handling_alias.ipynb

## Key Datasets
QueriesWithTemplate.csv -- all fields needed for the project.
	:attribute sqlid: SQL entry ID, from SDSS
	:attribute sessionid: session ID of the query entry, from SDSS
	:attribute localrank: order based on the start time of the query within its session 
	:attribute statementid: statement ID, from SDSS
	:attribute statement: raw query statement, from SDSS
	:attribute error: if 0, error-free, from SDSS
	:attribute processed: replaced with <ID> and <NUM>
	:attribute qdict: processed query in dictionary
	:attribute template_v2: processed template to handle edge cases
	:attribute template: templatified processed query
	:attribute tid: template ID

	:version sdss/model_data/sampled: 25% random sampling, from SDSS
	:version sdss/model_data/full: 100%, from SDSS
	:version sqlshare/model_data: 100%, from SQLShare
