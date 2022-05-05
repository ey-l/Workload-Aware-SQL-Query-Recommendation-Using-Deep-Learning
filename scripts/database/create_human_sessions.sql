-- Take the output of get_session_label.sql

-- ========================================================
-- Author: Eugenie
-- Description: 
-- 		Identify weblog records' class with agentstringID
-- 		Identify session class based on weblog records
-- 		Keep the hitID, sessionID, and class
-- ========================================================

DROP TABLE IF EXISTS HUMANSESSION CASCADE;

CREATE TABLE HUMANSESSION (
	sqlID 			BIGINT NOT NULL,
	theTime         TIMESTAMP NOT NULL,
	dbname          CHAR(32),
	statementID     BIGINT NOT NULL,
	rankinsession   INT NOT NULL,
	error           INT NOT NULL,
	sessionID 		BIGINT NOT NULL,
	class 			VARCHAR(10) NOT NULL
);

WITH SELOGS AS (	-- To get the startTime and limitTime of each session
	SELECT se.sessionID, se.class, selog.ID, selog.rankinsession
	FROM SESSIONLABEL se
	INNER JOIN SESSIONLOG selog ON se.sessionID = selog.sessionID
		AND se.class = 'BROWSER'
		AND selog.type = 1 -- Webhit
)
INSERT INTO HUMANSESSION
SELECT sql.sqlID, sql.theTime, sql.dbname, sql.statementID, sl.rankinsession, sql.error, sl.sessionID, sl.class
FROM SQLLOG sql
INNER JOIN SELOGS sl ON sl.ID = sql.sqlID -- Join with sessionlog 
	
