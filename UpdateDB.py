__author__ = 'Elahe'


import sqlite3 as lite
import numpy as np
import ephem
import pandas as pd

def update(Schedule):

    con = lite.connect('FBDE.db')
    cur = con.cursor()

    #TODO avoid overwrite
    t_start = Schedule[0]['ephemDate']
    t_end   = Schedule[-1]['ephemDate']

    N_visits = np.count_nonzero(Schedule['Field_id'])
    N_DD     = 0  # number of deep drilling observations


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

        #detect deep drilling observation to be reflected in NightSummary
        try:
            if Field_id == Schedule[index+ 1]['Field_id'] and Field_id == Schedule[index+ 2]['Field_id']:
                N_DD += 3
        except:
            pass
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
    Avg_cost    = np.average(Schedule['Cost'])
    Avg_slew_t  = np.average(Schedule['Slew_t'])
    Avg_alt     = np.average(Schedule['Alt'])
    Avg_ha      = np.average(Schedule['HA'])


    cur.execute('INSERT INTO NightSummary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (Night_count, t_start, t_end,  Initial_field, N_visits, N_DD, N_triple, N_double,
                     N_single, N_per_hour, Avg_cost, Avg_slew_t/ephem.second, Avg_alt, Avg_ha))


    ''' Update the FIELDS STATISTICS db'''

    for index, id in enumerate(Schedule['Field_id']):
        cur.execute('SELECT * FROM FieldsStatistics WHERE ID = ?',(id,))
        field_row = cur.fetchone()

        previous_Third_last_visit  = field_row[5]
        previous_Second_last_visit = field_row[6]
        previous_Last_visit        = field_row[7]
        previous_N_visit           = field_row[8]
        previous_Last_visit_u      = field_row[9]
        previous_N_visit_u         = field_row[10]
        previous_Last_visit_g      = field_row[11]
        previous_N_visit_g         = field_row[12]
        previous_Last_visit_r      = field_row[13]
        previous_N_visit_r         = field_row[14]
        previous_Last_visit_i      = field_row[15]
        previous_N_visit_i         = field_row[16]
        previous_Last_visit_z      = field_row[17]
        previous_N_visit_z         = field_row[18]
        previous_Last_visit_y      = field_row[19]
        previous_N_visit_y         = field_row[20]
        previous_Coadded_depth     = field_row[21]
        previous_Avg_cost          = field_row[22]
        previous_Avg_slew_t        = field_row[23]
        previous_Avg_alt           = field_row[24]
        previous_Avg_ha            = field_row[25]

        Fourth_last_visit = previous_Third_last_visit
        Third_last_visit  = previous_Second_last_visit
        Second_last_visit = previous_Last_visit
        Last_visit        = Schedule[index]['ephemDate']
        N_visit           = previous_N_visit + 1
        if Schedule[index]['Filter'] == 'u':
            Last_visit_u        = Schedule[index]['ephemDate']
            N_visit_u           = previous_N_visit_u + 1
        else:
            Last_visit_u        = previous_Last_visit_u
            N_visit_u           = previous_N_visit_u

        if Schedule[index]['Filter'] == 'g':
            Last_visit_g        = Schedule[index]['ephemDate']
            N_visit_g           = previous_N_visit_g + 1
        else:
            Last_visit_g        = previous_Last_visit_g
            N_visit_g           = previous_N_visit_g

        if Schedule[index]['Filter'] == 'r':
            Last_visit_r        = Schedule[index]['ephemDate']
            N_visit_r           = previous_N_visit_r + 1
        else:
            Last_visit_r        = previous_Last_visit_r
            N_visit_r           = previous_N_visit_r

        if Schedule[index]['Filter'] == 'i':
            Last_visit_i        = Schedule[index]['ephemDate']
            N_visit_i           = previous_N_visit_i + 1
        else:
            Last_visit_i        = previous_Last_visit_i
            N_visit_i           = previous_N_visit_i

        if Schedule[index]['Filter'] == 'z':
            Last_visit_z        = Schedule[index]['ephemDate']
            N_visit_z           = previous_N_visit_z + 1
        else:
            Last_visit_z        = previous_Last_visit_z
            N_visit_z           = previous_N_visit_z

        if Schedule[index]['Filter'] == 'y':
            Last_visit_y        = Schedule[index]['ephemDate']
            N_visit_y           = previous_N_visit_y + 1
        else:
            Last_visit_y        = previous_Last_visit_y
            N_visit_y           = previous_N_visit_y

        Coadded_depth     = previous_Coadded_depth + 0 # temporarily
        Avg_cost          = ((previous_Avg_cost * previous_N_visit) + Schedule[index]['Cost'])/N_visit
        Avg_slew_t        = ((previous_Avg_slew_t * previous_N_visit) + Schedule[index]['Slew_t'])/N_visit
        Avg_slew_t        = Avg_slew_t/ephem.second
        Avg_alt           = ((previous_Avg_alt * previous_N_visit) + Schedule[index]['Alt'])/N_visit
        Avg_ha            = ((previous_Avg_ha * previous_N_visit) + Schedule[index]['HA'])/N_visit

        cur.execute('UPDATE FieldsStatistics SET '
                        'Fourth_last_visit = ?, '
                        'Third_last_visit  = ?, '
                        'Second_last_visit = ?, '
                        'Last_visit        = ?, '
                        'N_visit           = ?, '
                        'Last_visit_u      = ?, '
                        'N_visit_u         = ?, '
                        'Last_visit_g      = ?, '
                        'N_visit_g         = ?, '
                        'Last_visit_r      = ?, '
                        'N_visit_r         = ?, '
                        'Last_visit_i      = ?, '
                        'N_visit_i         = ?, '
                        'Last_visit_z      = ?, '
                        'N_visit_z         = ?, '
                        'Last_visit_y      = ?, '
                        'N_visit_y         = ?, '
                        'Coadded_depth     = ?, '
                        'Avg_cost          = ?, '
                        'Avg_slew_t        = ?, '
                        'Avg_alt           = ?, '
                        'Avg_ha            = ? WHERE ID = ?',
                        (Fourth_last_visit, Third_last_visit, Second_last_visit, Last_visit, N_visit, Last_visit_u, N_visit_u,
                         Last_visit_g, N_visit_g, Last_visit_r, N_visit_r, Last_visit_i, N_visit_i, Last_visit_z, N_visit_z,
                         Last_visit_y, N_visit_y, Coadded_depth, Avg_cost, Avg_slew_t, Avg_alt, Avg_ha, id))

    ''' update the Filter Statistics db'''
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    cur.execute('SELECT * FROM FilterStatistics')
    all_data_frame = pd.DataFrame(cur.fetchall(), columns=['ID', 'Name', 'Fourth_last_visit','Third_last_visit',
                                                          'Second_last_visit', 'Last_visit', 'N_visit', 'N_usage', 'Avg_n_visit_in_a_row',
                                                          'Avg_alt', 'Avg_brightness', 'Avg_seeing'])

    all_data = all_data_frame.copy()
    visit_in_a_row = [0,0,0,0,0,0]
    visit_in_a_row_flag = False
    for i,f in enumerate(Schedule['Filter']):
        index = filters.index(f)
        all_data['Fourth_last_visit'][index] = all_data['Third_last_visit'][index]
        all_data['Third_last_visit'][index]  = all_data['Second_last_visit'][index]
        all_data['Second_last_visit'][index] = all_data['Last_visit'][index]
        all_data['Last_visit'][index]        = Schedule[i]['ephemDate']
        all_data['N_visit'][index]          += 1
        visit_in_a_row[index] += 1
        try:
            if Schedule[i+1]['Filter'] != f: # filter change happens at the next visit
                all_data['N_usage'][index] += 1
                all_data['Avg_n_visit_in_a_row'][index] = float((all_data['N_usage'][index] -1)*all_data['Avg_n_visit_in_a_row'][index] + visit_in_a_row[index])/all_data['N_usage'][index]
                visit_in_a_row[index] = 0
        except: # last visit of the episode
            all_data['N_usage'][index] += 1
            all_data['Avg_n_visit_in_a_row'][index] = float((all_data['N_usage'][index] -1)*all_data['Avg_n_visit_in_a_row'][index] + visit_in_a_row[index])/all_data['N_usage'][index]

        all_data['Avg_alt'][index] = ((all_data['N_visit'][index] -1)*all_data['Avg_alt'][index] + Schedule[i]['Alt'])/ all_data['N_visit'][index]
        all_data['Avg_brightness'][index] = ((all_data['N_visit'][index] -1)*all_data['Avg_brightness'][index] + Schedule[i]['Sky_bri'])/ all_data['N_visit'][index]


    for index,f in enumerate(filters):
        cur.execute('UPDATE FilterStatistics SET '
                    'Fourth_last_visit    = ?,'
                    'Third_last_visit     = ?,'
                    'Second_last_visit    = ?,'
                    'Last_visit           = ?,'
                    'N_visit              = ?,'
                    'N_usage              = ?,'
                    'Avg_n_visit_in_a_row = ?,'
                    'Avg_alt              = ?,'
                    'Avg_brightness       = ?,'
                    'Avg_seeing           = ? WHERE Name = ?',
                    (all_data['Fourth_last_visit'][index],
                    all_data['Third_last_visit'][index],
                    all_data['Second_last_visit'][index],
                    all_data['Last_visit'][index],
                    all_data['N_visit'][index],
                    all_data['N_usage'][index],
                    all_data['Avg_n_visit_in_a_row'][index],
                    all_data['Avg_alt'][index],
                    all_data['Avg_brightness'][index],
                    all_data['Avg_seeing'][index],
                    f))

    con.commit()


'''

Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.

n_nights = 1 # number of the nights to be scheduled starting from 1st Jan. 2021

Date_start = ephem.Date(ephem.Date('2020/12/31 12:00:00.00')) # times are in UT

for i in range(n_nights):
    Date = ephem.Date(Date_start + i) # times are in UT

    Schedule = np.load("Output/Schedule{}.npy".format(i + 1))
    update(Schedule)
'''

