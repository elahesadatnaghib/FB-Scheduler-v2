__author__ = 'Elahe'

import ephem
import numpy as np
import sqlite3 as lite
import os

def creatFBDE():
    # Delete previous database
    try:
        os.remove('FBDE.db')
    except:
        pass


    inf = 1e10
    eps = 1e-10

    ''' Connect to the FBDE data base '''
    con = lite.connect('FBDE.db')
    cur = con.cursor()

    cur.execute('CREATE TABLE Schedule('
                'Visit_count INTEGER PRIMARY KEY, '
                'Field_id INTEGER, '
                'ephemDate REAL, '
                'Filter INTEGER, '
                'n_ton INTEGER, '
                'n_previous INEGER, '
                'Cost REAL, '
                'Slew_t REAL, '
                't_since_v_ton REAL,'
                't_since_v_prev REAL,'
                'Alt REAL, '
                'HA REAL, '
                't_to_invis REAL, '
                'Sky_bri REAL, '
                'Temp_coverage REAL,'
                'F1 REAL,'
                'F2 REAL,'
                'F3 REAL,'
                'F4 REAL,'
                'F5 REAL,'
                'F6 REAL,'
                'F7 REAL)')

    cur.execute('CREATE TABLE NightSummary('
                'Night_count INTEGER PRIMARY KEY, '
                'T_start REAL, '
                'T_end REAL, '
                'Initial_field, '
                'N_visits INTEGER, '
                'N_DDcosmology INTEGER,'
                'N_triple INTEGER, '
                'N_double INTEGER, '
                'N_single INTEGER, '
                'N_per_hour REAL, '
                'Avg_cost REAL, '
                'Avg_slew_t REAL, '
                'Avg_alt REAL, '
                'Avg_ha REAL)')

    cur.execute('CREATE TABLE FieldsStatistics('
                'ID INTEGER, '
                'Dec REAL, '
                'RA REAL, '
                'Label TEXT,'
                'Fourth_last_visit REAL, '
                'Third_last_visit REAL, '
                'Second_last_visit REAL, '
                'Last_visit REAL, '
                'N_visit INTEGER, '
                'Last_visit_u REAL, '
                'N_visit_u INTEGER, '
                'Last_visit_g REAL, '
                'N_visit_g INTEGER, '
                'Last_visit_r REAL, '
                'N_visit_r INTEGER, '
                'Last_visit_i REAL, '
                'N_visit_i INTEGER, '
                'Last_visit_z REAL, '
                'N_visit_z INTEGER, '
                'Last_visit_y REAL, '
                'N_visit_y INTEGER, '
                'Coadded_depth REAL, '
                'Avg_cost REAL, '
                'Avg_slew_t REAL, '
                'Avg_alt REAL, '
                'Avg_ha REAL)')

    cur.execute('CREATE INDEX Idx1 ON FieldsStatistics(ID)')

    cur.execute('CREATE TABLE FilterStatistics('
                'ID INTEGER,'
                'Name TEXT,'
                'Fourth_last_visit REAL,'
                'Third_last_visit REAL,'
                'Second_last_visit REAL,'
                'Last_visit REAL,'
                'N_visit INTEGER,'
                'N_usage INTEGER,'
                'Avg_n_visit_in_a_row Integer,'
                'Avg_alt REAL,'
                'Avg_brightness REAL,'
                'Avg_seeing REAL)')

    cur.execute('CREATE TABLE IF NOT EXISTS FieldData ('
                'rowid INTEGER PRIMARY KEY, '
                'nightid INTEGER, '
                'ephemDate REAL, '
                'fieldid INTEGER, '
                'altitude REAL, '
                'azimuth REAL, '
                'hourangle REAL, '
                'moonseparation REAL,'
                'visible BOOL,'
                'covered BOOL,'
                'brightness REAL)')

    cur.execute('CREATE INDEX Idx2 ON FieldData(ephemDate)')

    cur.execute('CREATE TABLE ModelParam('
                'rowid INTEGER,'
                'Name TEXT, '
                'Value REAL)')

    all_fields = np.loadtxt("NightDataInLIS/Constants/fieldID.lis", dtype = "i4, f8, f8, S10")

    for field in all_fields:
        ID                = int(field[0])
        RA                = field[1]
        Dec               = field[2]
        Label             = field[3]
        Fourth_last_visit = -inf
        Third_last_visit  = -inf
        Second_last_visit = -inf
        Last_visit        = -inf
        N_visit           = 0
        Last_visit_u        = -inf
        N_visit_u           = 0
        Last_visit_g        = -inf
        N_visit_g           = 0
        Last_visit_r        = -inf
        N_visit_r           = 0
        Last_visit_i        = -inf
        N_visit_i           = 0
        Last_visit_z        = -inf
        N_visit_z           = 0
        Last_visit_y        = -inf
        N_visit_y           = 0
        Coadded_depth     = 0
        Avg_cost          = 0
        Avg_slew_t        = 0
        Avg_alt           = 0
        Avg_ha            = 0


        cur.execute('INSERT INTO FieldsStatistics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                     (ID, RA, Dec, Label, Fourth_last_visit, Third_last_visit, Second_last_visit,
                                      Last_visit, N_visit, Last_visit_u, N_visit_u, Last_visit_r, N_visit_r, Last_visit_i,
                                      N_visit_i, Last_visit_g, N_visit_g, Last_visit_z, N_visit_z, Last_visit_y, N_visit_y,
                                      Coadded_depth, Avg_cost, Avg_slew_t, Avg_alt, Avg_ha))

    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    for index, f in enumerate(filters):
        ID   = index +1
        Name = f
        Fourth_last_visit    = -inf
        Third_last_visit     = -inf
        Second_last_visit    = -inf
        Last_visit           = -inf
        N_visit              = 0
        N_usage              = 0
        Avg_n_visit_in_a_row = 0
        Avg_alt              = 0
        Avg_brightness       = 0
        Avg_seeing           = 0

        cur.execute('INSERT INTO FilterStatistics VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (ID, Name, Fourth_last_visit, Third_last_visit, Second_last_visit, Last_visit, N_visit, N_usage, Avg_n_visit_in_a_row,
                     Avg_alt, Avg_brightness, Avg_seeing))



    param_list = [('inf', inf),
                  ('eps', eps),
                  ('t_exposure', 32 * ephem.second),
                  ('visit_w1', 15 * ephem.minute),
                  ('visit_w2', 60 * ephem.minute),
                  ('max_n_night', 3),
                  ('data_t_interval', 10 * ephem.minute),
                  ('air_mass_lim', 1.4),
                  ('filter_change_t', 2 * ephem.minute)]

    rowid = 0
    for param in param_list:
        rowid +=1
        cur.execute('INSERT INTO ModelParam VALUES (?,?,?)', (rowid, param[0], param[1]))

    con.commit()
