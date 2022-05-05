-- Get session classes

-- ========================================================
-- Author: Eugenie
-- Description: 
-- 		Identify weblog records' class with agentstringID
-- 		Identify session class based on weblog records
-- 		Keep the hitID, sessionID, and class
-- ========================================================

DROP TABLE IF EXISTS SESSIONCLASS CASCADE;

CREATE TABLE SESSIONCLASS (
	hitID BIGINT NOT NULL,
	sessionID BIGINT NOT NULL,
	class VARCHAR(10) NOT NULL
);

WITH SELOGS AS (	-- To get the startTime and limitTime of each session
	SELECT se.sessionID, se.startTime, se.limitTime, selog.ID, selog.theTime
	FROM SESSION se
	INNER JOIN SESSIONLOG selog ON se.sessionID = selog.sessionID
		-- AND se.startTime < to_timestamp('2012-09-01 00:00:00', 'yyyy-mm-dd hh24:mi:ss') -- Get a smaller dataset for testing
		AND selog.type = 0 -- Webhit
		AND selog.theTime >= se.startTime -- Time constraints since session's PK is (sessionID, startTime, limitTime)
		AND selog.theTime <= se.limitTime
), AGENTSTR AS (
	SELECT was.agentStringID, wa.class
	FROM WEBAGENTSTRING was
	INNER JOIN WEBAGENT wa ON wa.agentID = was.agentID 
	-- AND wa.class NOT LIKE '%BOT%' -- Get only agent strings that are not "BOT" to speed up
)
INSERT INTO SESSIONCLASS
SELECT w.hitID, s.sessionID, astr.class
FROM ((WEBLOG w
	INNER JOIN SELOGS s ON s.ID = w.hitID -- Join with sessionlog 
	AND w.theTime >= s.startTime	-- Time constraints since session's PK is (sessionID, startTime, limitTime)
	AND w.theTime <= s.limitTime)
	INNER JOIN AGENTSTR astr ON w.agentStringID = astr.agentStringID); -- Join with agentstr to get webhit agent class

