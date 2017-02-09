__author__ = 'Elahe'


import sqlite3 as lite
import csv
import numpy as np
import ephem
import  NightDataGenerator as ndg


''' Connect to the FBDE data base '''
def DBreadNwrite(key, Date, Site, **keyword_parameters):
    '''**keyword_parameters
    sessionID
    '''
    if key == 'w':
        FBDEcon = lite.connect('MafFBDE.db')
        FBDEcur = FBDEcon.cursor()

        Watch    = np.load("Output/Watch{}.npy".format(int(ephem.julian_date(Date))))
        Schedule = np.load("Output/Schedule{}.npy".format(int(ephem.julian_date(Date))))
        Schedule['ephemDate'] += 15019.5
        Summary  = np.load("Output/Summary{}.npy".format(int(ephem.julian_date(Date))))
        Summary[0] += 15019.5; Summary[1] += 15019.5
        Conf     = np.load('NightDataInLIS/Config{}.npy'.format(int(ephem.julian_date(Date))))
        # 3 by n_fields matrix of ID, RA, Dec
        all_fields = np.loadtxt("NightDataInLIS/Constants/fieldID.lis", dtype = np.dtype([('id', 'i4'), ('ra', 'f8'), ('dec','f8')]))
        N_visits = np.count_nonzero(Schedule['Field_id'])

        sessionID = 0; sessionUser = 0; sessionHost = 0; sessionDate = 0; version = 0; runComment = 0
        if('sessionID' in keyword_parameters):
            sessionID = keyword_parameters['sessionID']
        if('sessionUser' in keyword_parameters):
            sessionUser = keyword_parameters['sessionUser']
        if('sessionHost' in keyword_parameters):
            sessionHost = keyword_parameters['sessionHost']
        if('sessionDate' in keyword_parameters):
            sessionDate = keyword_parameters['sessionDate']
        if('version' in keyword_parameters):
            version = keyword_parameters['version']
        if('runComment' in keyword_parameters):
            runComment = keyword_parameters['runComment']


        FBDEcur.execute('CREATE TABLE IF NOT EXISTS Config ('
                        'configID INTEGER PRIMARY KEY, '
                        'moduleName TEXT, '
                        'paramIndex INTEGER, '
                        'paramName TEXT, '
                        'paramValue TEXT, '
                        'comment TEXT, '
                        'Session_sessionID INTEGER, '
                        'nonPropID INTEGER)')

        with open('NightDataInLIS/Constants/conf.dmp','rb') as fin:
            dr = csv.DictReader(fin) # comma is default delimiter
            to_db = [(i['configID'], i['moduleName'], i['paramIndex'], i['paramName'], i['paramValue'], i['comment'], sessionID, i['nonPropID']) for i in dr]
        FBDEcur.executemany("INSERT INTO Config (configID, moduleName, paramIndex, paramName, paramValue, comment, Session_sessionID, nonPropID) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", to_db)
        FBDEcon.commit()


        FBDEcur.execute('CREATE TABLE IF NOT EXISTS Session ('
                        'sessionID INTEGER PRIMARY KEY, '
                        'sessionUser TEXT, '
                        'sessionHost TEXT, '
                        'sessionDate TEXT, '
                        'version TEXT, '
                        'runComment TEXT)')

        try:
            FBDEcur.execute('INSERT INTO Session VALUES (?, ?, ?, ?, ?, ?)',
                         (sessionID, sessionUser, sessionHost, sessionDate, version, runComment))
        except:
            pass


        FBDEcur.execute('CREATE TABLE IF NOT EXISTS ObsHistory ('
                        'obsHistID INTEGER PRIMARY KEY, '
                        'Session_sessionID INTEGER, '
                        'filter TEXT, '
                        'expDate INTEGER, '
                        'expMJD REAL, '
                        'night INTEGER, '
                        'visitTime REAL, '
                        'visitExpTime REAL, '
                        'finRank REAL, '
                        'finSeeing REAL, '
                        'transparency REAL, '
                        'airmass REAL, '
                        'vSkyBright REAL, '
                        'filtSkyBrightness REAL, '
                        'rotSkyPos REAL, '
                        'lst REAL, '
                        'altitude REAL, '
                        'azimuth REAL, '
                        'dist2Moon REAL, '
                        'solarElong REAL, '
                        'moonRA REAL, '
                        'moonDec REAL, '
                        'moonAlt REAL, '
                        'moonAZ REAL, '
                        'moonPhase REAL, '
                        'sunAlt REAL, '
                        'sunAZ REAL, '
                        'phaseAngle REAL, '
                        'rScatter REAL, '
                        'mieScatter REAL, '
                        'moonIllum REAL, '
                        'moonBright REAL, '
                        'darkBright REAL, '
                        'rawSeeing REAL, '
                        'wind REAL, '
                        'humidity REAL, '
                        'fiveSigmaDepth REAL, '
                        'ditheredRA REAL, '
                        'ditheredDec REAL, '
                        'Field_fieldID INTEGER)')

        obsHistID = 0; Session_sessionID = sessionID; filter = 0; expDate  = 0; expMJD  = 0; night   = 0
        visitTime = float(Conf['visitTime'])
        visitExpTime = float(Conf['visitExpTime'])
        finRank = 0; finSeeing   = 0; transparency   = 0; airmass   = 0
        vSkyBright   = 0; filtSkyBrightness   = 0
        rotSkyPos   = 0; lst  = 0; altitude  = 0; azimuth  = 0; dist2Moon   = 0; solarElong  = 0; moonRA  = 0
        moonDec  = 0; moonAlt  = 0; moonAZ  = 0; moonPhase  = 0; sunAlt  = 0; sunAZ  = 0; phaseAngle  = 0
        rScatter  = 0; mieScatter  = 0; moonIllum  = 0; moonBright  = 0; darkBright  = 0; rawSeeing  = 0; wind  = 0
        humidity  = 0; fiveSigmaDepth  = 0; ditheredRA  = 0; ditheredDec  = 0; Field_fieldID  = 0


        FBDEcur.execute('CREATE TABLE IF NOT EXISTS SlewHistory ('
                        'slewID INTEGER PRIMARY KEY, '
                        'slewCount INTEGER, '
                        'startDate REAL, '
                        'endDate REAL, '
                        'slewTime REAL, '
                        'slewDist REAL, '
                        'ObsHistory_obsHistID INTEGER, '
                        'ObsHistory_Session_sessionID INTEGER)')

        slewID = 0; slewCount = 0; startDate = 0; endDate = 0; slewTime = 0; slewDist = 0; ObsHistory_obsHistID = 0
        ObsHistory_Session_sessionID = 0
        try:
            FBDEcur.execute('SELECT * FROM ObsHistory ORDER BY SlewHistory DESC LIMIT 1')
            last_row_sch = FBDEcur.fetchone()
            slewCount = last_row_sch[0]
        except:
            pass

        try:
            FBDEcur.execute('SELECT * FROM ObsHistory ORDER BY obsHistID DESC LIMIT 1')
            last_row_sch = FBDEcur.fetchone()
            obsHistID = last_row_sch[0]
        except:
            pass

        try:
            FBDEcur.execute('SELECT * FROM ObsHistory ORDER BY night DESC LIMIT 1')
            last_row_ns = FBDEcur.fetchone()
            night = last_row_ns[0]
        except:
            pass
        night  += 1

        source = ephem.FixedBody()
        prev_field = ephem.FixedBody()
        moon = ephem.Moon()
        sun = ephem.Sun()
        for index in range(N_visits):
            obsHistID       += 1
            slewCount       += 1
            Field_fieldID    = Schedule[index]['Field_id']
            expMJD           = Schedule[index]['ephemDate'] - visitTime * ephem.second
            expDate          = (expMJD - 59560 + 15019.5)/ ephem.second
            filter           = Schedule[index]['Filter']
            ### Astro parameters
            Site.date = expMJD - 15019.5
            eq = ephem.Equatorial(np.radians(all_fields['ra'][Field_fieldID -1]), np.radians(all_fields['dec'][Field_fieldID -1]))
            source._ra = eq.ra
            source._dec = eq.dec
            source._epoch = eq.epoch
            source.compute(Site)
            altitude = source.alt
            azimuth = source.az
            airmass = ndg.secz(altitude)
            moon.compute(Site)
            dist2Moon = ephem.separation(moon,source)
            moonRA    = moon.ra
            moonDec   = moon.dec
            moonAlt   = moonAlt
            moonAZ    = moon.az
            moonPhase = moon.phase
            sun.compute(Site)
            sunAlt    = sun.alt
            sunAZ     =sun.az
            try:
                slewDist = ephem.separation(ssource, prev_field)
            except:
                slewDist = 0
            prev_field = source

            n_ton            = Schedule[index]['n_ton']
            n_last           = Schedule[index]['n_last']
            Cost             = Schedule[index]['Cost']
            t_since_v_ton = Schedule[index]['t_since_v_ton']
            t_since_v_last= Schedule[index]['t_since_v_last']
            Alt           = Schedule[index]['Alt']
            HA            = Schedule[index]['HA']
            t_to_invis    = Schedule[index]['t_to_invis']
            Sky_bri       = Schedule[index]['Sky_bri']
            Temp_coverage = Schedule[index]['Temp_coverage']
            slewTime           = Schedule[index]['Slew_t']
            startDate     = expMJD + visitTime * ephem.second
            endDate       = startDate + slewTime

            FBDEcur.execute('INSERT INTO ObsHistory VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,'
                                                           '?, ?, ?, ?, ?, ?, ?, ?, ?, ?,'
                                                           '?, ?, ?, ?, ?, ?, ?, ?, ?, ?,'
                                                           '?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                            (obsHistID, Session_sessionID, filter, expDate, expMJD, night, visitTime, visitExpTime,
                             finRank, finSeeing, transparency, airmass, vSkyBright, filtSkyBrightness, rotSkyPos,
                             lst, altitude, azimuth, dist2Moon, solarElong, moonRA, moonDec, moonAlt, moonAZ,
                             moonPhase,sunAlt, sunAZ, phaseAngle, rScatter, mieScatter, moonIllum, moonBright,
                             darkBright, rawSeeing, wind, humidity, fiveSigmaDepth, ditheredRA, ditheredDec, Field_fieldID))

            FBDEcur.execute('INSERT INTO SlewHistory VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                            (slewCount, slewCount, startDate, endDate, slewTime, slewDist,
                             ObsHistory_obsHistID, ObsHistory_Session_sessionID))

        FBDEcur.execute('CREATE TABLE Field ('
                        'fieldID INTEGER PRIMARY KEY, '
                        'fieldFov REAL, '
                        'fieldRA REAL, '
                        'fieldDec REAL, '
                        'fieldGL REAL, '
                        'fieldGB REAL, '
                        'fieldEL REAL, '
                        'fieldEB REAL)')

        fieldID = all_fields['id']; fieldRA = all_fields['ra']; fieldDec = all_fields['dec']
        fieldFov= 0; fieldGL = 0; fieldGB = 0; fieldEL = 0; fieldEB = 0

        for id,ra,dec in zip(fieldID, fieldRA, fieldDec):
            FBDEcur.execute('INSERT INTO Field VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        (int(id), ra, dec, fieldFov, fieldGL, fieldGB, fieldEL, fieldEB))



        FBDEcur.execute('CREATE TABLE Proposal ('
                        'propID INTEGER PRIMARY KEY, '
                        'propConf TEXT, '
                        'propName TEXT, '
                        'objectID INTEGER, '
                        'objectHost TEXT, '
                        'Session_sessionID INTEGER)')

        propID = 0; propConf = 0; propName = 0; objectID = 0; objectHost = 0; Session_sessionID =0;
        if('propID' in keyword_parameters):
            propID = keyword_parameters['propID']
        if('propConf' in keyword_parameters):
            propConf = keyword_parameters['propConf']
        if('propName' in keyword_parameters):
            propName = keyword_parameters['propName']
        if('objectID' in keyword_parameters):
            objectID = keyword_parameters['objectID']
        if('objectHost' in keyword_parameters):
            objectHost = keyword_parameters['objectHost']
        if('Session_sessionID' in keyword_parameters):
            Session_sessionID = keyword_parameters['Session_sessionID']

        FBDEcur.execute('INSERT INTO Proposal VALUES (?, ?, ?, ?, ?, ?)',
                        (propID, propConf, propName, objectID, objectHost, Session_sessionID))

        FBDEcur.execute('CREATE TABLE SeqHistory ('
                        'sequenceID INTEGER PRIMARY KEY, '
                        'startDate INTEGER, '
                        'expDate INTEGER, '
                        'seqnNum INTEGER, '
                        'completion REAL, '
                        'reqEvents INTEGER, '
                        'actualEvents INTEGER, '
                        'endStatus INTEGER, '
                        'parent_sequenceID INTEGER, '
                        'Field_fieldID INTEGER, '
                        'Session_sessionID INTEGER, '
                        'Proposal_propID INTEGER)')


        sequenceID = 0; startDate =0; expDate = 0; seqnNum = 0; completion =0; reqEvents = 0; actualEvents =0
        endStatus = 0;  parent_sequenceID = 0;  Field_fieldID = 0; Session_sessionID = Session_sessionID; Proposal_propID = 0

        FBDEcur.execute('INSERT INTO SeqHistory VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (sequenceID, startDate, expDate, seqnNum, completion, reqEvents, actualEvents,
                         endStatus, parent_sequenceID, Field_fieldID, Session_sessionID, Proposal_propID))

        FBDEcur.execute('CREATE TABLE SlewActivities ('
                        'slewActivityID INTEGER PRIMARY KEY, '
                        'activity TEXT, '
                        'actDelay REAL, '
                        'inCriticalPath TEXT, '
                        'SlewHistory_slewID INTEGER)')
        slewActivityID = 0; activity = 0; actDelay = 0; inCriticalPath = 0; SlewHistory_slewID = 0
        FBDEcur.execute('INSERT INTO SlewActivities VALUES (?, ?, ?, ?, ?)',(slewActivityID, activity, actDelay, inCriticalPath, SlewHistory_slewID))



        FBDEcur.execute('CREATE TABLE SlewState ('
                        'slewIniStatID INTEGER PRIMARY KEY, '
                        'slewStateDate REAL, '
                        'tra REAL, '
                        'tdec REAL, '
                        'tracking TEXT, '
                        'alt REAL, '
                        'az REAL, '
                        'pa REAL, '
                        'domAlt REAL, '
                        'domAz REAL, '
                        'telAlt REAL, '
                        'telAz REAL, '
                        'rotTelPos REAL, '
                        'filter TEXT, '
                        'state INTEGER, '
                        'SlewHistory_slewID INTEGER)')

        FBDEcur.execute('CREATE TABLE SlewMaxSpeeds ('
                        'slewMaxSpeedID INTEGER PRIMARY KEY, '
                        'domAltSpd REAL, '
                        'domAzSpd REAL, '
                        'telAltSpd REAL, '
                        'telAzSpd REAL, '
                        'rotSpd REAL, '
                        'SlewHistory_slewID INTEGER)')

        FBDEcur.execute('CREATE TABLE TimeHistory ('
                        'timeHistID INTEGER PRIMARY KEY, '
                        'date INTEGER, '
                        'mjd REAL, '
                        'night INTEGER, '
                        'event INTEGER, '
                        'Session_sessionID INTEGER)')

        FBDEcur.execute('CREATE TABLE ObsHistory_Proposal ('
                        'obsHistory_propID INTEGER PRIMARY KEY, '
                        'Proposal_propID INTEGER, '
                        'propRank REAL, '
                        'ObsHistory_obsHistID INTEGER, '
                        'ObsHistory_Session_sessionID INTEGER)')

        FBDEcur.execute('CREATE TABLE Cloud ('
                        'cloudID INTEGER PRIMARY KEY, '
                        'c_date INTEGER, '
                        'cloud REAL)')

        FBDEcur.execute('CREATE TABLE Seeing ('
                        'seeingID INTEGER PRIMARY KEY, '
                        's_date INTEGER, '
                        'seeing REAL)')

        FBDEcur.execute('CREATE TABLE Log ('
                        'logID INTEGER PRIMARY KEY, '
                        'log_name TEXT, '
                        'log_value TEXT, '
                        'Session_sessionID INTEGER)')

        FBDEcur.execute('CREATE TABLE Config_File ('
                        'config_fileID INTEGER PRIMARY KEY, '
                        'filename TEXT, '
                        'data TEXT, '
                        'Session_sessionID INTEGER)')

        FBDEcur.execute('CREATE TABLE Proposal_Field ('
                        'proposal_field_id INTEGER PRIMARY KEY, '
                        'Session_sessionID INTEGER, '
                        'Proposal_propID INTEGER, '
                        'Field_fieldID INTEGER)')

        FBDEcur.execute('CREATE TABLE SeqHistory_ObsHistory ('
                        'seqhistory_obsHistID INTEGER PRIMARY KEY, '
                        'SeqHistory_sequenceID INTEGER, '
                        'ObsHistory_obsHistID INTEGER, '
                        'ObsHistory_Session_sessionID INTEGER)')

        FBDEcur.execute('CREATE TABLE MissedHistory ('
                        'missedHistID INTEGER PRIMARY KEY, '
                        'Session_sessionID INTEGER, '
                        'filter TEXT, expDate INTEGER, '
                        'expMJD REAL, '
                        'night INTEGER, '
                        'lst REAL, '
                        'Field_fieldID INTEGER)')

        FBDEcur.execute('CREATE TABLE SeqHistory_MissedHistory ('
                        'seqhistory_missedHistID INTEGER PRIMARY KEY, '
                        'SeqHistory_sequenceID INTEGER, '
                        'MissedHistory_missedHistID INTEGER, '
                        'MissedHistory_Session_sessionID INTEGER)')

        FBDEcur.execute('CREATE TABLE Summary ('
                        'obsHistID INTEGER, '
                        'sessionID INTEGER, '
                        'propID INTEGER, '
                        'fieldID INTEGER, '
                        'fieldRA REAL, '
                        'fieldDec REAL, '
                        'filter TEXT, '
                        'expDate INTEGER, '
                        'expMJD REAL, '
                        'night INTEGER, '
                        'visitTime REAL, '
                        'visitExpTime REAL, '
                        'finRank REAL, '
                        'finSeeing REAL, '
                        'transparency REAL, '
                        'airmass REAL, '
                        'vSkyBright REAL, '
                        'filtSkyBrightness REAL, '
                        'rotSkyPos REAL, '
                        'lst REAL, '
                        'altitude REAL, '
                        'azimuth REAL, '
                        'dist2Moon REAL, '
                        'solarElong REAL, '
                        'moonRA REAL, '
                        'moonDec REAL, '
                        'moonAlt REAL, '
                        'moonAZ REAL, '
                        'moonPhase REAL, '
                        'sunAlt REAL, '
                        'sunAz REAL, '
                        'phaseAngle REAL, '
                        'rScatter REAL, '
                        'mieScatter REAL, '
                        'moonIllum REAL, '
                        'moonBright REAL, '
                        'darkBright REAL, '
                        'rawSeeing REAL, '
                        'wind REAL, '
                        'humidity REAL, '
                        'slewDist REAL, '
                        'slewTime REAL, '
                        'fiveSigmaDepth REAL, '
                        'ditheredRA REAL, '
                        'ditheredDec REAL)')

    if key == 'r':

        return


    return

Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.

Date = ephem.Date('2016/09/01 12:00:00.00') # times are in UT
DBreadNwrite('w', Date, Site, sessionID = 1, sessionUser= 'Elahe')
