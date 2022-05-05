
select * from humansession where LOWER(dbname) like 'bestdr%' limit 10;
select count(distinct sessionid) from humansession where LOWER(dbname) like 'bestdr%';

Copy (select * from humansession where LOWER(dbname) like 'bestdr%') To 'F:\data\processed\\test\humansession.csv' With CSV DELIMITER ',' HEADER;

-- Get maximum starttime
select max(session.starttime) from session;