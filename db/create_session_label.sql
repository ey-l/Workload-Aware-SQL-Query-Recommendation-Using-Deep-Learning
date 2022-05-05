-- Take the output of get_session_majority.sql

-- ========================================================
-- Author: Eugenie
-- Description: 
--      Get the session class label based on the majority vote
-- 		and the 'BOT' webhit of the session.
-- 		Identify a session 'BOT' if there is at least one 'BOT' webhit
-- ========================================================

DROP TABLE IF EXISTS SESSIONLABEL CASCADE;

CREATE TABLE SESSIONLABEL(
	sessionID BIGINT NOT NULL,
	class VARCHAR(10) NOT NULL
);

INSERT INTO SESSIONLABEL
SELECT sm.sessionID, sm.class
FROM SESSIONMAJ sm
WHERE sm.sessionID NOT IN 
	(SELECT DISTINCT sessionID
    FROM SESSIONCLASS
    WHERE class = 'BOT') -- Keep the majority vote class label if absend
UNION
SELECT DISTINCT a.sessionID, a.class -- Get sessions with at least one BOT webhit
FROM SESSIONCLASS a
WHERE a.class = 'BOT';