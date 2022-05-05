-- Create the sdssweblogs database in postgresql 12
/*
DROP TABLE IF EXISTS SESSION CASCADE;
DROP TABLE IF EXISTS SESSIONLOG CASCADE;
DROP TABLE IF EXISTS WEBAGENT CASCADE;
DROP TABLE IF EXISTS WEBAGENTSTRING CASCADE;
DROP TABLE IF EXISTS WEBLOG CASCADE;
DROP TABLE IF EXISTS SQLLOG CASCADE;
DROP TABLE IF EXISTS SQLSTATEMENT;
*/

SET datestyle = "ISO, MDY";


CREATE TABLE SESSION (
    sessionID       BIGINT NOT NULL,
    firsteventID    BIGINT NOT NULL,
    sessionipID     INT NOT NULL,
    startTime       TIMESTAMP NOT NULL,
    limitTime       TIMESTAMP NOT NULL,
    seconds         INT NOT NULL,
    events          INT NOT NULL,
    webhits         INT NOT NULL,
    sqlqueries      INT NOT NULL,
    sky1hits        INT NOT NULL,
    sky1pageviews   INT NOT NULL,
    PRIMARY KEY (sessionID, firsteventID, startTime, limitTime)
  );

COPY session FROM 'F:\\data\\db\\session.csv' WITH DELIMITER AS ',';


CREATE TABLE SESSIONLOG (
    sessionID       BIGINT NOT NULL,
    rankinsession   INT NOT NULL,
    ipID            INT NOT NULL,
    theTime         TIMESTAMP NOT NULL,
    type            INT NOT NULL,
    ID              BIGINT NOT NULL,
    templateID      BIGINT NOT NULL,
    PRIMARY KEY (sessionID, rankinsession)
  );

-- COPY sessionlog FROM 'F:\\data\\db\\sessionlog.csv' WITH DELIMITER AS ','; -- Doesn't work because the file is too large
COPY sessionlog from PROGRAM 'cmd /c "type F:\\data\\db\\sessionlog.csv"' WITH DELIMITER AS ',';
-- The trick here is to run cmd in a single command mode, with /c and telling it to type out the file in the double quotations

CREATE TABLE WEBLOG (
    hitID           BIGINT NOT NULL,
    yy              SMALLINT NOT NULL,
    mm              SMALLINT NOT NULL,
    dd              SMALLINT NOT NULL,
    hh              SMALLINT NOT NULL,
    mi              SMALLINT NOT NULL,
    ss              SMALLINT NOT NULL,
    theTime         TIMESTAMP NOT NULL,
    logID           INT NOT NULL,
    clientipID      BIGINT,
    opID            INT,
    stemID          BIGINT,
    paramsID        BIGINT,
    error           INT NOT NULL,
    agentStringID   BIGINT NOT NULL,
    bytesOut        BIGINT NOT NULL,
    bytesIn         BIGINT NOT NULL,
    elapsed         REAL NOT NULL,
    isvisible       BYTEA NOT NULL,
    pageView        BYTEA NOT NULL,
    SOAP            BYTEA NOT NULL,
    SQL             BYTEA NOT NULL,
    skyserver       SMALLINT NOT NULL,
    studyperiod     SMALLINT NOT NULL,
    includeflag     BYTEA NOT NULL,
    PRIMARY KEY (hitID, yy, mm, dd, hh, mi, ss)
  );

-- COPY weblog FROM 'F:\\data\\db\\weblog.csv' WITH DELIMITER AS ',';
COPY weblog from PROGRAM 'cmd /c "type F:\\data\\db\\weblog.csv"' WITH DELIMITER AS ',';


CREATE TABLE SQLLOG (
    sqlID           BIGINT NOT NULL,
    yy              SMALLINT NOT NULL,
    mm              SMALLINT NOT NULL,
    dd              SMALLINT NOT NULL,
    hh              SMALLINT NOT NULL,
    mi              SMALLINT NOT NULL,
    ss              SMALLINT NOT NULL,
    theTime         TIMESTAMP NOT NULL,
    logID           INT,
    clientipID      BIGINT,
    requestor       CHAR(32),
    server          CHAR(32),
    dbname          CHAR(32),
    access          CHAR(32),
    elapsed         REAL,
    busy            REAL,
    rows            BIGINT,
    statementID     BIGINT NOT NULL,
    error           INT NOT NULL,
    errorMessageID  INT NOT NULL,
    PRIMARY KEY (sqlID, yy, mm, dd, hh, mi, ss)
  );

-- COPY sqllog FROM 'F:\\data\\db\\sqllog.csv' WITH DELIMITER AS ',';
COPY sqllog from PROGRAM 'cmd /c "type F:\\data\\db\\sqllog.csv"' WITH DELIMITER AS ',';


CREATE TABLE WEBAGENT (
    agentID         INT NOT NULL,
    agentName       VARCHAR(100) NOT NULL,
    agentSubCategory VARCHAR(100),
    agentPattern    VARCHAR(100) NOT NULL,
    class           VARCHAR(10) NOT NULL,
    hits            INT NOT NULL,
    pageviews       INT NOT NULL,
    sky1hits        INT NOT NULL,
    sky1pageviews   INT NOT NULL,
    PRIMARY KEY (agentID)
  );

COPY webagent FROM 'F:\\data\\db\\webagent.csv' WITH DELIMITER AS ',';


CREATE TABLE WEBAGENTSTRING (
    agentStringID   BIGINT NOT NULL,
    agentID         INT NOT NULL,
    PRIMARY KEY (agentStringID)
  );

COPY webagentstring FROM 'F:\\data\\db\\webagentstring.csv' WITH DELIMITER AS ',';

-- ALTER TABLE WEBAGENTSTRING ADD FOREIGN KEY (agentID) REFERENCES WEBAGENT(agentID);


CREATE TABLE SQLSTATEMENT (
    statementID   BIGINT NOT NULL,
    statement     VARCHAR,
    hits          BIGINT NOT NULL,
    templateID    BIGINT NOT NULL,
    studyperiod   SMALLINT NOT NULL,
    PRIMARY KEY (statementID)
  );

COPY sqlstatement FROM PROGRAM 'cmd /c "type F:\\data\\sdssweblogs\\sqlstatement.csv"' WITH DELIMITER AS ',' CSV HEADER;

