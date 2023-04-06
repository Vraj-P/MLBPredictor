DROP TABLE IF EXISTS `NumAwardsPlayers`;
CREATE TABLE NumAwardsPlayers AS (
	SELECT 
		People.playerID, 
        COUNT(awardID) AS numAwards 
	FROM (
		People LEFT OUTER JOIN AwardsPlayers 
		ON People.playerID = AwardsPlayers.playerID
	) GROUP BY playerID
);

DROP TABLE IF EXISTS `InductedPlayers`;
CREATE TABLE InductedPlayers AS (
	SELECT 
		playerID, 
        inducted 
	FROM HallOfFame
	WHERE inducted = 'Y'
);

DROP TABLE IF EXISTS `NominatedPlayers`;
CREATE TABLE NominatedPlayers AS (
	SELECT DISTINCT
		playerID, 
        inducted 
	FROM HallOfFame
	WHERE playerID NOT IN (
		SELECT 
		playerID
		FROM HallOfFame
		WHERE inducted = 'Y'
	)
);

DROP TABLE IF EXISTS `HallOfFameSimplified`;
CREATE TABLE HallOfFameSimplified AS (
	SELECT 
		playerID,
		inducted 
	FROM NominatedPlayers
	UNION
    SELECT
		playerID,
        inducted
	FROM InductedPlayers
);

DROP TABLE IF EXISTS `HallOfFameALL`;
CREATE TABLE HallOfFameALL AS (
	SELECT 
		People.playerID, 
		inducted
	FROM (
		People LEFT OUTER JOIN HallOfFameSimplified 
		ON People.playerID = HallOfFameSimplified.playerID
	)
);
    
DROP TABLE IF EXISTS `BattingAll`;
CREATE TABLE BattingAll AS (
	SELECT 
		People.playerID, 
		SUM(H) AS hitsBatter, 
		SUM(R) AS runsBatter, 
		SUM(HR) AS homerunsBatter, 
		SUM(SB) AS stolenBases 
	FROM (
		People LEFT OUTER JOIN Batting 
		ON People.playerID = Batting.playerID
	) 
	GROUP BY People.playerID
);
  
DROP TABLE IF EXISTS `FieldingAll`;
CREATE TABLE FieldingAll AS (
	SELECT 
		People.playerID, 
		SUM(PO) AS putouts, 
		SUM(A) AS assists
	FROM (
		People LEFT OUTER JOIN Fielding 
		ON People.playerID = Fielding.playerID
	) 
	GROUP BY People.playerID
);
    
DROP TABLE IF EXISTS `PitchingAll`;
CREATE TABLE PitchingAll AS (    
	SELECT 
		People.playerID, 
		SUM(SHO) AS shutouts, 
		SUM(SV) AS saves, 
		SUM(H) AS hitsPitcher, 
		SUM(ER) AS earnedRuns, 
		SUM(HR) AS homerunsPitcher, 
		SUM(SO) AS strikeouts 
	FROM (
		People LEFT OUTER JOIN Pitching 
		ON People.playerID = Pitching.playerID
	)
	GROUP BY People.playerID
);

DROP TABLE IF EXISTS `TaskATable`;
CREATE TABLE TaskATable AS (    
	SELECT
		playerID,
        numAwards,
        hitsBatter,
        runsBatter,
        homerunsBatter,
        stolenBases,
        putouts,
        assists,
        shutouts,
        saves,
        hitsPitcher,
        earnedRuns,
        homerunsPitcher,
        strikeouts,
        inducted AS class
	FROM (
		NumAwardsPlayers 
		NATURAL JOIN 
		BattingAll
		NATURAL JOIN 
		FieldingAll
		NATURAL JOIN 
		PitchingAll
		NATURAL JOIN 
		HallOfFameALL 
	)
);

DROP TABLE IF EXISTS `TaskBTable`;
CREATE TABLE TaskBTable AS (    
	SELECT
		* 
	FROM TaskATable
    WHERE class = 'Y' OR class = 'N'
);

UPDATE TaskATable 
SET class = CASE class 
WHEN 'Y' THEN TRUE
WHEN 'N' THEN TRUE
ELSE FALSE END;

UPDATE TaskATable SET numAwards = 0 WHERE numAwards IS NULL;
UPDATE TaskATable SET hitsBatter = 0 WHERE hitsBatter IS NULL;
UPDATE TaskATable SET runsBatter = 0 WHERE runsBatter IS NULL;
UPDATE TaskATable SET homerunsBatter = 0 WHERE homerunsBatter IS NULL;
UPDATE TaskATable SET stolenBases = 0 WHERE stolenBases IS NULL;
UPDATE TaskATable SET putouts = 0 WHERE putouts IS NULL;
UPDATE TaskATable SET assists = 0 WHERE assists IS NULL;
UPDATE TaskATable SET shutouts = 0 WHERE shutouts IS NULL;
UPDATE TaskATable SET saves = 0 WHERE saves IS NULL;
UPDATE TaskATable SET hitsPitcher = 0 WHERE hitsPitcher IS NULL;
UPDATE TaskATable SET earnedRuns = 0 WHERE earnedRuns IS NULL;
UPDATE TaskATable SET homerunsPitcher = 0 WHERE homerunsPitcher IS NULL;
UPDATE TaskATable SET strikeouts = 0 WHERE strikeouts IS NULL;

UPDATE TaskBTable 
SET class = CASE class WHEN 'Y' THEN TRUE
ELSE FALSE END;

UPDATE TaskBTable SET numAwards = 0 WHERE numAwards IS NULL;
UPDATE TaskBTable SET hitsBatter = 0 WHERE hitsBatter IS NULL;
UPDATE TaskBTable SET runsBatter = 0 WHERE runsBatter IS NULL;
UPDATE TaskBTable SET homerunsBatter = 0 WHERE homerunsBatter IS NULL;
UPDATE TaskBTable SET stolenBases = 0 WHERE stolenBases IS NULL;
UPDATE TaskBTable SET putouts = 0 WHERE putouts IS NULL;
UPDATE TaskBTable SET assists = 0 WHERE assists IS NULL;
UPDATE TaskBTable SET shutouts = 0 WHERE shutouts IS NULL;
UPDATE TaskBTable SET saves = 0 WHERE saves IS NULL;
UPDATE TaskBTable SET hitsPitcher = 0 WHERE hitsPitcher IS NULL;
UPDATE TaskBTable SET earnedRuns = 0 WHERE earnedRuns IS NULL;
UPDATE TaskBTable SET homerunsPitcher = 0 WHERE homerunsPitcher IS NULL;
UPDATE TaskBTable SET strikeouts = 0 WHERE strikeouts IS NULL;
