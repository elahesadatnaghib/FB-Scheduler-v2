__author__ = 'Elahe'

import numpy as np
import ephem
import FBDE
import time
import os.path

#my modules
from UpdateDB import update
from Graphics import visualize



Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.


#F_weight        = [ 10.47816163 , 18.51547084  , 5.62295409 , 10.93592489,  12.2933596, 11.79469227   ,7.26407901]#582.165391198
#F_weight        = [  9.03840438 , 14.29264535  , 6.0003242 ,  12.97667936,  17.48667836, 2.77361073 , 14.82248513]#601.517852964
F_weight        = [  9.41574316 , 14.82260668 , 10.88215625 , 13.99381687 , 18.69126356, 2.77361073 , 14.17433157]#621.847279558
# F1: slew time cost 0~2
# F2: night urgency -1~1
# F3: overall urgency 0~1
# F4: altitude cost 0~1
# F5: hour angle cost 0~1
# F6: co-added depth cost 0~1
# F7: normalized brightness 0~1
preferences     = [1,1,4,0,3,10]


n_nights = 1 # number of the nights to be scheduled starting from 1st Jan. 2021

s = time.time()

Date_start = ephem.Date('2015/6/28 12:00:00.00') # times are in UT


for i in range(n_nights):
    Date = ephem.Date(Date_start + 2) # times are in UT

    # create scheduler and import data
    t0 = time.time()
    scheduler = FBDE.Scheduler(Date, Site, F_weight,gray_train = True,  custom_period = 0.1)
    t1 = time.time()
    print('\nData of the {} imported in {} sec'.format(Date, t1 - t0))

    # schedule
    scheduler.schedule()
    print(scheduler.eval_performance(preferences))
    t2 = time.time()
    print('\nScheduling of the {} finished in {} sec'.format(Date, t2 - t1))

    '''
    # update the database
    Schedule = np.load("Output/Schedule{}.npy".format(i + 1))
    update(Schedule)
    t3 = time.time()
    print('\nDatabase for {} updated in {} sec'.format(Date, t3 - t2))

    # create animation
    FPS = 10            # Frame per second
    Steps = 100          # Simulation steps
    MP4_quality = 300   # MP4 size and quality

    PlotID = 1        # 1 for one Plot, 2 for including covering pattern
    visualize(Date, PlotID ,FPS, Steps, MP4_quality, 'Visualizations/LSST1plot{}.mp4'.format(i + 1), showClouds= True)
    t4 = time.time()
    print('\nVisualization for {} created in {} sec'.format(Date, t4 - t3))

    '''

print('\n \nTotal elapsed time: {} minutes'.format((time.time() - s)/60))

