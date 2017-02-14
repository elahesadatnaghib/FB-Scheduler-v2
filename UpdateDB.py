__author__ = 'Elahe'


import sqlite3 as lite
import numpy as np
import ephem

def DBreadNwrite(Schedule):

    con = lite.connect('FBDE.db')
    cur = con.cursor()

    #TODO avoid overwrite
    t_start = Schedule[0]['ephemDate']
    t_end   = Schedule[-1]['ephemDate']

    N_visits = np.count_nonzero(Schedule['Field_id'])



    ''' Update the SCHEDULE db'''
    # Import last row of the data base
    try:
        cur.execute('SELECT * FROM Schedule ORDER BY Visit_count DESC LIMIT 1')
        last_row_sch = cur.fetchone()
        Visit_count = last_row_sch[0]
    except:
        Visit_count = 0

    for index in range(N_visits):
        Visit_count  += 1
        Field_id      = Schedule[index]['Field_id']
        ephemDate     = Schedule[index]['ephemDate']
        Filter        = Schedule[index]['Filter']
        n_ton         = Schedule[index]['n_ton']
        n_last        = Schedule[index]['n_last']
        Cost          = Schedule[index]['Cost']
        Slew_t        = Schedule[index]['Slew_t']
        t_since_v_ton = Schedule[index]['t_since_v_ton']
        t_since_v_last= Schedule[index]['t_since_v_last']
        Alt           = Schedule[index]['Alt']
        HA            = Schedule[index]['HA']
        t_to_invis    = Schedule[index]['t_to_invis']
        Sky_bri       = Schedule[index]['Sky_bri']
        Temp_coverage = Schedule[index]['Temp_coverage']
        F1            = Schedule[index]['F1']
        F2            = Schedule[index]['F2']
        F3            = Schedule[index]['F3']
        F4            = Schedule[index]['F4']
        F5            = Schedule[index]['F5']
        F6            = Schedule[index]['F6']
        F7            = Schedule[index]['F7']

        cur.execute('INSERT INTO Schedule VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                             (Visit_count, Field_id, ephemDate, Filter, n_ton, n_last, Cost, Slew_t/ephem.second,
                              t_since_v_ton, t_since_v_last, Alt, HA, t_to_invis, Sky_bri, Temp_coverage,
                              F1, F2, F3, F4, F5, F6, F7))


    ''' Update the NIGHT SUMMARY db'''
    # Import last row of the data base
    try:
        cur.execute('SELECT * FROM NightSummary ORDER BY Night_count DESC LIMIT 1')
        last_row_ns = cur.fetchone()
        Night_count = last_row_ns[0]
    except:
        Night_count = 0

    Night_count  += 1
    Initial_field = int(Schedule['Field_id'][0])
    N_visits      = N_visits

    u, c           = np.unique(Schedule['Field_id'], return_counts=True)
    unique, counts = np.unique(c, return_counts=True)
    try:
        N_triple    = counts[unique == 3][0]
    except:
        N_triple    = 0
    try:
        N_double    = counts[unique == 2][0]
    except:
        N_double    = 0
    try:
        N_single    = counts[unique == 1][0]
    except:
        N_single    = 0

    N_per_hour  = N_visits * ephem.hour/ (t_end - t_start)
    Avg_cost    = np.average(Schedule[0:N_visits]['Cost'])
    Avg_slew_t  = np.average(Schedule[0:N_visits]['Slew_t'])
    Avg_alt     = np.average(Schedule[0:N_visits]['Alt'])
    Avg_ha      = np.average(Schedule[0:N_visits]['HA'])

    cur.execute('INSERT INTO NightSummary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (Night_count, t_start, t_end,  Initial_field, N_visits, N_triple, N_double,
                     N_single, N_per_hour, Avg_cost, Avg_slew_t/ephem.second, Avg_alt, Avg_ha))


    ''' Update the FIELDS STATISTICS db'''

    for index, id in enumerate(Schedule['Field_id']):
        cur.execute('SELECT * FROM FieldsStatistics WHERE ID = ?',(id,))
        field_row = cur.fetchone()

        previous_Third_last_visit  = field_row[4]
        previous_Second_last_visit = field_row[5]
        previous_Last_visit        = field_row[6]
        previous_N_visit           = field_row[7]
        previous_Coadded_depth     = field_row[8]
        previous_Avg_cost          = field_row[9]
        previous_Avg_slew_t        = field_row[10]
        previous_Avg_alt           = field_row[11]
        previous_Avg_ha            = field_row[12]

        Fourth_last_visit = previous_Third_last_visit
        Third_last_visit  = previous_Second_last_visit
        Second_last_visit = previous_Last_visit
        Last_visit        = Schedule[index]['ephemDate']
        N_visit           = previous_N_visit + 1
        Coadded_depth     = previous_Coadded_depth + 0 # temporarily
        Avg_cost          = ((previous_Avg_cost * previous_N_visit) + Schedule[index]['Cost'])/N_visit
        Avg_slew_t        = ((previous_Avg_slew_t * previous_N_visit) + Schedule[index]['Slew_t'])/N_visit
        Avg_alt           = ((previous_Avg_alt * previous_N_visit) + Schedule[index]['Alt'])/N_visit
        Avg_ha            = ((previous_Avg_ha * previous_N_visit) + Schedule[index]['HA'])/N_visit

        cur.execute('UPDATE FieldsStatistics SET '
                        'Fourth_last_visit = ?, '
                        'Third_last_visit  = ?, '
                        'Second_last_visit = ?, '
                        'Last_visit        = ?, '
                        'N_visit           = ?, '
                        'Coadded_depth     = ?, '
                        'Avg_cost          = ?, '
                        'Avg_slew_t        = ?, '
                        'Avg_alt           = ?, '
                        'Avg_ha            = ? WHERE ID = ?',
                        (Fourth_last_visit, Third_last_visit, Second_last_visit, Last_visit, N_visit, Coadded_depth, Avg_cost, Avg_slew_t/ephem.second, Avg_alt, Avg_ha, id))


    con.commit()



Schedule = np.load("Output/Schedule{}.npy".format(44195.5556134))
DBreadNwrite(Schedule)