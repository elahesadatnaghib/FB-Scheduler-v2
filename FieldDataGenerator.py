__author__ = 'Elahe'

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

def Field_data(site, end_d):

    con = lite.connect('FBDE.db')
    cur = con.cursor()

    # grab model parameters
    cur.execute('SELECT Value FROM ModelParam')
    input        = cur.fetchall()
    dt           = input[6][0]
    airmassLimit = input[7][0]


    #Check if there is some data generated before
    try:
        cur.execute('SELECT * FROM FieldData ORDER BY rowid DESC LIMIT 1')
        last_row_sch = cur.fetchone()
        rowid_moments= last_row_sch[0]
        nightid      = last_row_sch[1]
        last_t = last_row_sch[2]
        start_d = int(last_t)+1 # the next night

    except:
        rowid_moments= 0
        nightid      = 0
        start_d = ephem.Date('2020/12/31 12:00:00.00')


    # initialize sources
    moon = ephem.Moon()
    fields = np.loadtxt("NightDataInLIS/Constants/fieldID.lis", dtype = "i4, f8, f8, S10")
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
        all_nights.append(start_d + i)

    # Main loops
    for night in all_nights:
        nightid += 1
        # define date and time interval for airmass, and airmass limits
        # times are in UT
        time_range = set_data_range(lsst, night, dt)
        for s_index,s in enumerate(source):
            temp = 0
            temp2=0.
            for t_index,t in enumerate(time_range):
                lsst.date = ephem.Date(t)
                moon.compute(lsst)
                ephemDate = float(t)
                rowid_moments += 1
                s.compute(lsst)
                altitude = s.alt
                #visible = bool(secz(altitude) < airmassLimit)
                visible  = bool(float(altitude) > 0.7751933733084387)
                #if s_index == 338:
                    #print(visible, t_index, float(altitude))
                azimuth = s.az
                hourang = (float(lsst.sidereal_time()) - float(s.ra))*12.0/np.pi
                if hourang > 12:
                    hourang = hourang - 24
                if hourang < -12:
                    hourang = hourang + 24
                moonsep = ephem.separation(moon,s)
                covered = False
                brightness = 0.

                cur.execute('INSERT INTO FieldData VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                            (rowid_moments, nightid, ephemDate, int(fields[s_index][0]), altitude, azimuth, hourang,
                             moonsep, visible, covered, brightness))

                if visible:
                    temp += 1
                temp2+=altitude

            #print(temp, temp2/len(time_range))

    con.commit()


'''

Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.

end_date        = ephem.Date('2020/12/31 12:00:00.00') + 2

Field_data(Site, end_date)

'''