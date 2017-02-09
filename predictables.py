import ephem
import numpy as np
from datetime import datetime
import sqlite3 as lite


def set_data_range(lsst, date, tint):
    '''Return numpy array of dates between astronomical twilight'''
    ss = set_time(ephem.Date(twilightEve(lsst, date)))
    sr = set_time(ephem.Date(twilightMorn(lsst, date)))
    return np.arange(ss, sr, tint)

def set_time(dtime):
    '''Ses time to rounded value'''
    y, m ,d, hh, mm, ss = dtime.tuple()
    mm = mm - (mm % 5)
    return ephem.Date(datetime(y, m , d, hh, mm, 5, 0))

def sunset(site, date):
    '''Sunset in UTC'''
    site.horizon = 0.
    sun = ephem.Sun()
    site.date = date
    return site.next_setting(sun)

def sunrise(site, date):
    '''Sunset in UTC'''
    site.horizon = 0.
    sun = ephem.Sun()
    site.date = date
    return site.next_rising(sun)

def twilightEve(site, date):
    '''Start of twilight in UTC'''
    site.horizon = "-18."
    sun = ephem.Sun()
    site.date = date
    return site.next_setting(sun)

def twilightMorn(site, date):
    '''End of twilight in UTC'''
    site.horizon = "-18."
    sun = ephem.Sun()
    site.date = date
    return site.next_rising(sun)

def secz(alt):
    '''Compute airmass'''
    if alt < ephem.degrees('03:00:00'):
        alt = ephem.degrees('03:00:00')
    sz = 1.0/np.sin(alt) - 1.0
    xp = 1.0 + sz*(0.9981833 - sz*(0.002875 + 0.0008083*sz))
    return xp

def effectiveTime(airmass, extinction_coefficient=0.11):
    '''Calculate the effective exposure time given an airmass'''
    t_exp = 30.
    extinction = 10**((extinction_coefficient*(airmass - 1.))/2.5)
    return t_exp/(extinction*extinction)

def Night_data(site, start_d, end_d, dt = 10 * ephem.minute, airmassLimit = 1.4):

    con = lite.connect('FBDE.db')
    cur = con.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS ByMoments ('
                        'rowid INTEGER PRIMARY KEY, '
                        'nightid INTEGER, '
                        'ephemdate REAL, '
                        'fieldid INTEGER, '
                        'altitude REAL, '
                        'azimuth REAL, '
                        'hourangle REAL, '
                        'moonseparation REAL)' )

    cur.execute('CREATE TABLE IF NOT EXISTS ByNights ('
                        'rowid INTEGER PRIMARY KEY, '
                        'nightid INTEGER, '
                        'nightdate REAL,'
                        'fieldid INTEGER, '
                        'effectiverise REAL, '
                        'effectiveset REAL, '
                        'nvisit REAL, '
                        'tlastvisit REAL)')

    #Check if there is some data generated before
    try:
        cur.execute('SELECT * FROM ByMoments ORDER BY rowid DESC LIMIT 1')
        last_row_sch = cur.fetchone()
        rowid_moments= last_row_sch[0]
        nightid      = last_row_sch[1]
        last_t = last_row_sch[2]
        start_d = int(last_t)+1 # the next night
        cur.execute('SELECT * FROM ByNights ORDER BY rowid DESC LIMIT 1')
        last_row_sch = cur.fetchone()
        rowid_nights = last_row_sch[0]
    except:
        rowid_moments= 0
        rowid_nights = 0
        nightid      = 0
        if start_d == 0:
            start_d = ephem.Date('2021/01/01 12:00:00.00')


    # initialize sources
    moon = ephem.Moon()
    fields = np.loadtxt('NightDataInLIS/Constants/fieldID.lis')
    source = []
    for f in fields:
        temp = ephem.FixedBody()
        eq = ephem.Equatorial(np.radians(f[1]), np.radians(f[2]))
        temp._ra = eq.ra
        temp._dec = eq.dec
        temp._epoch = eq.epoch
        source.append(temp)



    lsst = site
    # constructing an iterable of desired nights
    all_nights = []
    for i in range(int(end_d - start_d)):
        all_nights.append(start_d + 1)

    # Main loops
    for night in all_nights:
        nightid += 1
        # define date and time interval for airmass, and airmass limits
        # times are in UT
        hist = history_check(cur,night,len(fields))
        time_range = set_data_range(lsst, night, dt)
        for s_index,s in enumerate(source):
            amss_cstr = []  #to store all the airmasses for effective rise and set
            for t_index,t in enumerate(time_range):
                lsst.date = ephem.Date(t)
                moon.compute(lsst)
                ephemDate = float(t)
                rowid_moments += 1
                s.compute(lsst)
                altitude = s.alt
                airmass = secz(altitude)
                amss_cstr.append(airmass)
                azimuth = s.az
                hourang = (float(lsst.sidereal_time()) - float(s.ra))*12.0/np.pi
                if hourang > 12:
                    hourang = hourang - 24
                if hourang < -12:
                    hourang = hourang + 24
                moonsep = ephem.separation(moon,s)

                cur.execute('INSERT INTO ByMoments VALUES (?,?,?,?,?,?,?,?)',
                            (rowid_moments, nightid, ephemDate, fields[s_index][0], altitude, azimuth, hourang, moonsep))
            # inserting value
            rowid_nights += 1
            nightdate = int(float(time_range[0]))
            tot_n_visit = hist[s_index]['N_vis']
            t_last_visit = hist[s_index]['last_v']

            #apply airmass limits and extract rise and set times
            inRange = [i for i,a in enumerate(amss_cstr) if a < airmassLimit]
            if len(inRange):
                rising = inRange[0]
                setting = inRange[-1]
                effectiverise = float(time_range[rising])
                effectiveset  = float(time_range[setting])
            else:
                effectiverise = 0.
                effectiveset  = 0.

            cur.execute('INSERT INTO ByNights VALUES (?,?,?,?,?,?,?,?)',
                        (rowid_nights, nightid, nightdate, fields[s_index][0], effectiverise, effectiveset, tot_n_visit, t_last_visit))


    con.commit()


def history_check(cur, date, n_all_fields):
    try:
        cur.execute('SELECT T_start, T_end FROM NightSummary ORDER BY Night_count DESC LIMIT 1')
        mean_date = int(np.average(cur.fetchone()))
        if mean_date + 1 < int(date):
            print('There is no record of last night in the database, last available night is {}'.format(mean_date))
        cur.execute('SELECT Last_visit, N_visit FROM FieldsStatistics')
        hist = cur.fetchall()
    except: #when there is no history
        print('\n No database of previous observations found')
        hist = np.zeros(n_all_fields, dtype = [('last_v', np.float), ('N_vis', np.int)])
        hist['last_v'] = np.ones(n_all_fields) * 1e10

    return hist





Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.

start_date = ephem.Date('2021/01/01 12:00:00.00')
end_date = ephem.Date('2021/01/01 12:00:00.00') + 2

Night_data(Site,start_date, end_date)