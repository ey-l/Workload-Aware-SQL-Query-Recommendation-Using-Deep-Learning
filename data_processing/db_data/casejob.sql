-- Queries used to pull sqllog statements from casejob
-- The already-exported csv file is messed up due to the commas in the statements

-- Select yy=2009 from sqllogall
SELECT '"' + REPLACE(CAST(statement AS VARCHAR(8000)), '"', '""') + '"' into mydb.mytable FROM sqllogall 
WHERE yy=2009

-- Put sql statement into "" to make sure it gets exported into csv correctly
UPDATE mydb.mytable
SET statement = '"' + REPLACE(CAST(statement AS VARCHAR(8000)), '"', '""') + '"'