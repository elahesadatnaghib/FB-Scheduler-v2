import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import ephem
import sqlite3 as lite
from MultiProposalCalculations import *

# Altitude and Azimuth of a single field at t (JD) in rad
def Fields_local_coordinate(Field_ra, Field_dec, t, Site):

    # date and time
    Site.date = t
    curr_obj = ephem.FixedBody()
    curr_obj._ra = Field_ra * np.pi / 180
    curr_obj._dec = Field_dec * np.pi / 180
    curr_obj.compute(Site)
    altitude = curr_obj.alt
    azimuth = curr_obj.az
    return altitude, azimuth

def update_moon(t, Site) :
    Moon = ephem.Moon()
    Site.date = t
    Moon.compute(Site)
    X, Y = AltAz2XY(Moon.alt, Moon.az)
    r = Moon.size / 3600 * np.pi / 180  *2
    return X, Y, r, Moon.alt

def AltAz2XY(Alt, Az) :
    X = np.cos(Alt) * np.cos(Az) * -1
    Y = np.cos(Alt) * np.sin(Az)
    #Y = Alt * 2/ np.pi
    #X = Az / (2*np.pi)

    return Y, -1*X

def visualize(Date, PlotID = 1,FPS = 15,Steps = 20,MP4_quality = 300, Name = "LSST Scheduler Simulator.mp4", showClouds = False):

    # Import data
    All_Fields = np.loadtxt("NightDataInLIS/Constants/fieldID.lis", dtype = "i4, f8, f8, S10")
    N_Fields   = len(All_Fields)

    Site            = ephem.Observer()
    Site.lon        = -1.2320792
    Site.lat        = -0.517781017
    Site.elevation  = 2650
    Site.pressure   = 0.
    Site.horizon    = 0.

    if showClouds:
        Time_slots = np.loadtxt("NightDataInLIS/TimeSlots{}.lis".format(int(ephem.julian_date(Date))), unpack = True)
        All_Cloud_cover = np.loadtxt("NightDataInLIS/Clouds{}.lis".format(int(ephem.julian_date(Date))), unpack = True)


    #Initialize date and time
    lastN_start = float(Date) -1;   lastN_end = float(Date)
    toN_start = float(Date);        toN_end = float(Date) + 1

    #Connect to the History data base
    con = lite.connect('FBDE.db')
    cur = con.cursor()

    # Prepare to save in MP4 format
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='LSST Simulation', artist='Elahe', comment='Test')
    writer = FFMpegWriter(fps=FPS, metadata=metadata)

    # Initialize plot
    Fig = plt.figure()
    if PlotID == 1:
        ax = plt.subplot(111, axisbg = 'black')
    if PlotID == 2:
        ax = plt.subplot(211, axisbg = 'black')


    unobserved, Observed_lastN,\
    WFD, GP, NE, SE, DD,\
    Obseved_toN,\
    ToN_History_line,\
    uu,gg,rr,ii,zz,yy,\
    WFD_single,\
    last_10_History_line,\
    Horizon, airmass_horizon, S_Pole,\
    LSST,\
    Clouds\
        = ax.plot([], [], '*',[], [], '*',
                  [], [], '*',[], [], '*',[], [], '*', [], [], '*', [], [], '*', # proposals
                  [], [], '*',
                  [], [], '*',
                  [], [], '*',[], [], '*',[], [], '*', [], [], '*',[], [], '*',[], [], '*', #filters
                  [], [], '*', #WFD single
                  [], [], '-',
                  [], [], '-',[], [], '-',[], [], 'D',
                  [], [], 'o',
                  [], [], 'o')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', adjustable = 'box')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Coloring
    Horizon.set_color('white'); airmass_horizon.set_color('red')
    S_Pole.set_markersize(3);   S_Pole.set_markerfacecolor('red')
    star_size = 4

    unobserved.set_color('dimgray');        unobserved.set_markersize(star_size)
    Observed_lastN.set_color('blue');       Observed_lastN.set_markersize(star_size)
    Obseved_toN.set_color('chartreuse');    Obseved_toN.set_markersize(0)

    # filters
    uu.set_color('purple'); gg.set_color('green'); rr.set_color('red')
    ii.set_color('orange'); zz.set_color('pink');   yy.set_color('deeppink')

    # clouds
    Clouds.set_color('white');              Clouds.set_markersize(10)
    Clouds.set_alpha(0.2);                  Clouds.set_markeredgecolor(None)

    ToN_History_line.set_color('orange');   ToN_History_line.set_lw(.5)
    last_10_History_line.set_color('gray');  last_10_History_line.set_lw(.5)

    # proposals
    WFD.set_color('dimgray');               WFD.set_alpha(0.5)
    SE.set_color('green');                  SE.set_alpha(0.5)
    NE.set_color('blue');                   NE.set_alpha(0.5)
    GP.set_color('red');                    GP.set_alpha(0.5)
    DD.set_color('black');                  DD.set_alpha(0.5); DD.set_markersize(7)

    # WFD single visits
    WFD_single.set_color('dimgray');          WFD.set_alpha(0.5); WFD_single.set_markersize(10)

    LSST.set_color('red'); LSST.set_markersize(8)

    if PlotID == 2:
        freqAX = plt.subplot(212)
        cur.execute('SELECT N_visit, Last_visit, Second_last_visit, Third_last_visit, Fourth_last_visit From FieldsStatistics')
        row = cur.fetchall()
        N_visit     = [x[0] for x in row]
        Last_visit   = [x[1] for x in row]
        Second_last_visit = [x[2] for x in row]
        Third_last_visit  = [x[3] for x in row]
        Fourth_last_visit = [x[4] for x in row]

        initHistoricalcoverage = N_visit
        for index, id in enumerate(All_Fields[:,0]):
            if Last_visit[index] > toN_start:
                initHistoricalcoverage[index] -= 1
                if Second_last_visit[index] > toN_start:
                    initHistoricalcoverage[index] -= 1
                    if Third_last_visit > toN_start:
                        initHistoricalcoverage[index] -= 1



        covering, current_cover = freqAX.plot(All_Fields[:,0],initHistoricalcoverage,'-',[],[],'o')

        freqAX.set_xlim(0,N_Fields)
        freqAX.set_ylim(0,np.max(initHistoricalcoverage)+5)
        covering.set_color('chartreuse');   covering.set_markersize(2)
        current_cover.set_color('red');     current_cover.set_markersize(6)



    cur.execute('SELECT Night_count, T_start, T_end FROM NightSummary WHERE T_start BETWEEN (?) AND (?)',(toN_start, toN_end))
    row = cur.fetchone()
    vID = row[0]
    t_start = row[1]
    t_end   = row[2]
    t = t_start


    # Figure labels and fixed elements
    Phi = np.arange(0, 2* np.pi, 0.05)
    Horizon.set_data(1.01*np.cos(Phi), 1.01*np.sin(Phi))
    ax.text(-1.3, 0, 'West', color = 'white', fontsize = 7)
    ax.text(1.15, 0 ,'East', color = 'white', fontsize = 7)
    ax.text( 0, 1.1, 'North', color = 'white', fontsize = 7)
    airmass_horizon.set_data(np.cos(np.pi/4) * np.cos(Phi), np.cos(np.pi/4) *  np.sin(Phi))
    ax.text(-.3, 0.6, 'Acceptable airmass horizon', color = 'white', fontsize = 5, fontweight = 'bold')
    Alt, Az = Fields_local_coordinate(180, -90, t, Site)
    x, y = AltAz2XY(Alt,Az)
    S_Pole.set_data(x, y)
    ax.text(x+ .05, y, 'S-Pole', color = 'white', fontsize = 7)
    DD_indicator = ax.text(-1.4,1.3, 'Deep Drilling Observation', color = 'red', fontsize = 9, visible = False)
    WFD_indicator = ax.text(-1.4,1.3, 'White Fast Deep Observation', color = 'white', fontsize = 9, visible = False)
    GP_indicator = ax.text(-1.4,1.3, 'Galactic Plane Observation', color = 'white', fontsize = 9, visible = False)
    NES_indicator = ax.text(-1.4,1.3, 'Notrh Ecliptic Spur Observation', color = 'white', fontsize = 9, visible = False)
    SCP_indicator = ax.text(-1.4,1.3, 'South Celestial Pole Observation', color = 'white', fontsize = 9, visible = False)

    # Observed last night fields
    cur.execute('SELECT Field_id FROM Schedule WHERE ephemDate BETWEEN (?) AND (?)',(lastN_start, lastN_end))
    row = cur.fetchall()
    if row is not None:
        F1 = [x[0] for x in row]
    else:
        F1 = []

    # Tonight observation path
    cur.execute('SELECT Field_id, ephemDate, filter, Label FROM Schedule WHERE ephemDate BETWEEN (?) AND (?)',(toN_start, toN_end))
    row = cur.fetchall()
    if row[0][0] is not None:
        F2 = [x[0] for x in row]
        F2_timing = [x[1] for x in row]
        F2_filtering = [x[2] for x in row]
        F2_region    = [x[3] for x in row]
    else:
        F2 = []; F2_timing = []; F2_filtering = []; F2_region = []

    # Sky elements
    Moon = Circle((0, 0), 0, color = 'silver', zorder = 3)
    ax.add_patch(Moon)
    Moon_text = ax.text([], [], 'Moon', color = 'white', fontsize = 7)


    with writer.saving(Fig, Name, MP4_quality) :
        for t in np.linspace(t_start, t_end, num = Steps):


            # Find the index of the current time
            time_index = 0
            while t > F2_timing[time_index]:
                time_index += 1
            if showClouds:
                Slot_n = 0
                while t > Time_slots[Slot_n]:
                    Slot_n += 1

            visit_index = 0
            visited_field = 0
            visit_index_u = 0; visit_index_g = 0; visit_index_r = 0; visit_index_i = 0; visit_index_z = 0; visit_index_y = 0
            visit_index_wfd_s = 0
            visit_filter  = 'r'


            # Object fields: F1)Observed last night F2)Observed tonight F3)Unobserved F4)Covered by clouds
            F1_X = []; F1_Y = []; F2_X = []; F2_Y = []; F3_X = []; F3_Y = []; F4_X = []; F4_Y = []
            # Filter coloring for tonight observation
            U_X = []; U_Y = []; G_X = []; G_Y = []; R_X = []; R_Y = []; I_X = []; I_Y = []; Z_X = []; Z_Y = []; Y_X = []; Y_Y = []
            # Coloring different proposals
            WFD_X = []; WFD_Y = []; NE_X = []; NE_Y = []; SE_X = []; SE_Y = []; GP_X = []; GP_Y = []; DD_X = []; DD_Y = []
            #WFD single visits
            WFD_s_X = []; WFD_s_Y = []


            # F1  coordinate:
            for i in F1:
                Alt, Az = Fields_local_coordinate(All_Fields[i-1][1], All_Fields[i-1][2], t, Site)
                if Alt > 0:
                    X, Y    = AltAz2XY(Alt,Az)
                    F1_X.append(X); F1_Y.append(Y)

            # F2  coordinate:
            for i,tau,filter in zip(F2, F2_timing, F2_filtering):
                Alt, Az = Fields_local_coordinate(All_Fields[i-1][1], All_Fields[i-1][2], t, Site)
                if Alt > 0:
                    X, Y    = AltAz2XY(Alt,Az)
                    F2_X.append(X); F2_Y.append(Y)

                    if t >= tau:
                        visit_index = len(F2_X) -1
                        visited_field = i
                        visit_filter  = filter

                    # filter colored observation
                    if filter == 'u':
                        U_X.append(X); U_Y.append(Y)
                        if t >= tau:
                            visit_index_u = len(U_X) -1
                    elif filter == 'g':
                        G_X.append(X); G_Y.append(Y)
                        if t >= tau:
                            visit_index_g = len(G_Y) -1
                    elif filter == 'r':
                        R_X.append(X); R_Y.append(Y)
                        if t >= tau:
                            visit_index_r = len(R_Y) -1
                    elif filter == 'i':
                        I_X.append(X); I_Y.append(Y)
                        if t >= tau:
                            visit_index_i = len(I_Y) -1
                    elif filter == 'z':
                        Z_X.append(X); Z_Y.append(Y)
                        if t >= tau:
                            visit_index_z = len(Z_Y) -1
                    elif filter == 'y':
                        Y_X.append(X); Y_Y.append(Y)
                        if t >= tau:
                            visit_index_y = len(Y_Y) -1

            for i, tau in zip(F2, F2_timing):
                #white fast deep single visits
                if np.sum(np.asanyarray(F2[0:visit_index]) == i) == 1 and All_Fields[i -1][3] == 'WFD':
                    Alt, Az = Fields_local_coordinate(All_Fields[i-1][1], All_Fields[i-1][2], t, Site)
                    if Alt > 0:
                        X, Y    = AltAz2XY(Alt,Az)
                        WFD_s_X.append(X); WFD_s_Y.append(Y)
                        if t >= tau:
                            visit_index_wfd_s = len(WFD_s_Y) -1



            # F3  coordinate:
            for i in range(0,N_Fields):
                Alt, Az = Fields_local_coordinate(All_Fields[i][1], All_Fields[i][2], t, Site)
                if Alt > 0:
                    X, Y    = AltAz2XY(Alt,Az)
                    F3_X.append(X); F3_Y.append(Y)
                    if All_Fields[i][3] == 'DD':
                        DD_X.append(X); DD_Y.append(Y)
                    elif All_Fields[i][3] == 'WFD':
                        WFD_X.append(X); WFD_Y.append(Y)
                    elif All_Fields[i][3] == 'GP':
                        GP_X.append(X); GP_Y.append(Y)
                    elif All_Fields[i][3] == 'NES':
                        NE_X.append(X); NE_Y.append(Y)
                    elif All_Fields[i][3] == 'SCP':
                        SE_X.append(X); SE_Y.append(Y)

            # F4 coordinates
            if showClouds:
                for i in range(0,N_Fields):
                    if All_Cloud_cover[Slot_n,i] == -1 or All_Cloud_cover[Slot_n,i] == 1:# or All_Cloud_cover[Slot_n,i] == 2:
                        Alt, Az = Fields_local_coordinate(All_Fields[i][1], All_Fields[i][2], t, Site)
                    if Alt > 0:
                        X, Y    = AltAz2XY(Alt,Az)
                        F4_X.append(X); F4_Y.append(Y)


            # Update plot
            unobserved.set_data([F3_X,F3_Y])
            Observed_lastN.set_data([F1_X,F1_Y])
            Obseved_toN.set_data([F2_X[0:visit_index],F2_Y[0:visit_index]])

            # filters
            uu.set_data([U_X[0:visit_index_u],U_Y[0:visit_index_u]]); gg.set_data([G_X[0:visit_index_g],G_Y[0:visit_index_g]])
            rr.set_data([R_X[0:visit_index_r],R_Y[0:visit_index_r]]); ii.set_data([I_X[0:visit_index_i],I_Y[0:visit_index_i]])
            zz.set_data([Z_X[0:visit_index_z],Z_Y[0:visit_index_z]]); yy.set_data([Y_X[0:visit_index_y],Y_Y[0:visit_index_y]])

            ToN_History_line.set_data([F2_X[0:visit_index], F2_Y[0:visit_index]])
            last_10_History_line.set_data([F2_X[visit_index - 10: visit_index], F2_Y[visit_index - 10: visit_index]])

            # proposals
            WFD.set_data([WFD_X, WFD_Y]); DD.set_data([DD_X,DD_Y]); NE.set_data([NE_X, NE_Y]); SE.set_data([SE_X, SE_Y])
            GP.set_data([GP_X, GP_Y])

            # WFD singles
            WFD_single.set_data([WFD_s_X[0:visit_index_wfd_s], WFD_s_Y[0:visit_index_wfd_s]])

            # telescope position and color
            LSST.set_data([F2_X[visit_index],F2_Y[visit_index]])
            if visit_filter == 'u':
                LSST.set_color('purple')
            if visit_filter == 'g':
                LSST.set_color('green')
            if visit_filter == 'r':
                LSST.set_color('red')
            if visit_filter == 'i':
                LSST.set_color('orange')
            if visit_filter == 'z':
                LSST.set_color('pink')
            if visit_filter == 'y':
                LSST.set_color('deeppink')

            Clouds.set_data([F4_X,F4_Y])


            # Update Moon
            X, Y, r, alt = update_moon(t, Site)
            Moon.center = X, Y
            Moon.radius = r
            if alt > 0:
                #Moon.set_visible(True)
                Moon_text.set_visible(True)
                Moon_text.set_x(X+.002); Moon_text.set_y(Y+.002)
            else :
                Moon.set_visible(False)
                Moon_text.set_visible(False)

            #Update coverage
            if PlotID == 2:
                Historicalcoverage = np.zeros(N_Fields)
                for i,tau in zip(F2, F2_timing):
                    if tau <= t:
                        Historicalcoverage[i -1] += 1
                    else:
                        break
                tot = Historicalcoverage + initHistoricalcoverage
                current_cover.set_data(visited_field -1,tot[visited_field -1])
                covering.set_data(All_Fields[:,0], tot)

            #Update indicators of the proposal
            if F2_region[time_index]== 'DD':
                DD_indicator.set_visible(True)
            else:
                DD_indicator.set_visible(False)
            if F2_region[time_index]== 'WFD':
                WFD_indicator.set_visible(True)
            else:
                WFD_indicator.set_visible(False)
            if F2_region[time_index]== 'GP':
                GP_indicator.set_visible(True)
            else:
                GP_indicator.set_visible(False)
            if F2_region[time_index]== 'NES':
                NES_indicator.set_visible(True)
            else:
                NES_indicator.set_visible(False)
            if F2_region[time_index]== 'DD':
                SCP_indicator.set_visible(True)
            else:
                SCP_indicator.set_visible(False)


            #Observation statistics
            leg = plt.legend([Observed_lastN, Obseved_toN, uu, gg, rr, ii, zz, yy],
                       ['Visited last night', time_index,
                        'u filter', 'g filter', 'r filter', 'i filter', 'z filter', 'y filter'],
                             bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

            for l in leg.get_texts():
                l.set_fontsize(6)
            date = ephem.date(t)
            Fig.suptitle('Top view of the LSST site on {}, GMT'.format(date))


            '''
            # progress
            perc= int(100*(t - t_start)/(t_end - t_start))
            if perc <= 100:
                print('{} %'.format(perc))
            else:
                print('100 %')
            '''
            #Save current frame
            writer.grab_frame()



Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.

n_nights = 1 # number of the nights to be scheduled starting from 1st Jan. 2021

Date_start = ephem.Date('2015/6/28 12:00:00.00') # times are in UT

for i in range(n_nights):
    Date = ephem.Date(Date_start + i) # times are in UT

    # create animation
    FPS = 10            # Frame per second
    Steps = 50          # Simulation steps
    MP4_quality = 300   # MP4 size and quality

    PlotID = 1        # 1 for one Plot, 2 for including covering pattern
    visualize(Date, PlotID ,FPS, Steps, MP4_quality, 'Visualizations/LSST1plot_testtest{}.mp4'.format(i + 1), showClouds= True)

