__author__ = 'Elahe'


import sqlite3 as lite
import numpy as np
import ephem


''' Connect to the FBDE data base '''
def DBreadNwrite(episode_output):

    FBDEcon = lite.connect('FBDE.db')
    FBDEcur = FBDEcon.cursor()

    # avoid overwrite
    t_start = episode_output[0]['ephemDate']
    t_end   = episode_output[-1]['ephemDate']
    try:
        FBDEcur.execute('SELECT * FROM NightSummary ORDER BY Night_count DESC LIMIT 1')
        last_row_ns = FBDEcur.fetchone()
        t_start_db = last_row_ns[1]
        t_end_db   = last_row_ns[2]
        if int(t_start_db + t_end_db)/2 == int(Date):
            print('This night is already recorded in the database')
            return
    except:
        pass

    # avoid dropping a night
    try:
        if int(t_start_db + t_end_db)/2 < int(Date) -1:
            print('One or more night(s) are missing')
            return
        if int(t_start_db + t_end_db)/2 > int(Date) -1:
            print('Last recorded night is after the intended night')
            return
    except:
        pass





    Watch = np.load("Output/Watch{}.npy".format(int(ephem.julian_date(Date))))
    Schedule = np.load("Output/Schedule{}.npy".format(int(ephem.julian_date(Date))))
    Summary = np.load("Output/Summary{}.npy".format(int(ephem.julian_date(Date))))
    # 3 by n_fields matrix of ID, RA, Dec




    FBDEcon.commit()

    ''' Update the WATCH db'''
    # Import last row of the data base
    try:
        FBDEcur.execute('SELECT * FROM Watch ORDER BY Visit_count DESC LIMIT 1')
        last_row_sch = FBDEcur.fetchone()
        Visit_count = last_row_sch[0]
    except:
        Visit_count = 0

    for index in range(N_visits):
        Visit_count  += 1
        Field_id      = Watch[index]['Field_id']
        ephemDate     = Watch[index]['ephemDate']
        F1            = Watch[index]['F1']
        F2            = Watch[index]['F2']
        F3            = Watch[index]['F3']
        F4            = Watch[index]['F4']
        F5            = Watch[index]['F5']
        F6            = Watch[index]['F6']
        F7            = Watch[index]['F7']


        FBDEcur.execute('INSERT INTO Watch VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                             (Visit_count, Field_id, ephemDate, F1, F2, F3, F4, F5, F6, F7))
    FBDEcon.commit()



    return




'''
Date = ephem.Date('2016/09/01 12:00:00.00') # times are in UT
DBreadNwrite('w', Date)
'''