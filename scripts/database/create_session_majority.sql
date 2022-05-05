-- Take the output of get_session_classes.sql

-- ========================================================
-- Author: Eugenie
-- Description: 
--      Get the majority vote of the session class
-- ========================================================

DROP TABLE IF EXISTS SESSIONMAJ CASCADE;

CREATE TABLE SESSIONMAJ(
	sessionID BIGINT NOT NULL,
	class VARCHAR(10) NOT NULL
);

INSERT INTO SESSIONMAJ
SELECT DISTINCT
    t.sessionID, 
    t.class
FROM SESSIONCLASS t
JOIN
(
    SELECT
    sessionID,
    class,
    COUNT(sessionID) Ct
    FROM SESSIONCLASS
    GROUP BY sessionID, class
    ) Cont
ON t.sessionID = cont.sessionID
AND t.class = cont.class
JOIN
(
    SELECT
    a.sessionID,
    MAX(a.Ct) MaxCount
    FROM
    (
        SELECT
        sessionID,
        class,
        COUNT(sessionID) Ct
        FROM SESSIONCLASS
        GROUP BY sessionID, class
    ) a
    GROUP BY a.sessionID
) maxi ON t.sessionID = maxi.sessionID
AND cont.Ct = maxi.MaxCount