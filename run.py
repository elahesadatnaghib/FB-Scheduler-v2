__author__ = 'Elahe'

import numpy as np
import ephem
import FBDE
import time
import os.path

#my modules
import CreateDB




Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.

preferences     = [1,1,4,0,3,5] # mission objective
# objective function
#objective = preferences[0] * average cost * -1 +
#            preferences[1] * average slew time * -1 +
#            preferences[2] * average altitude  *  1 +
#            preferences[3] * No. of triple visits *  1 +
#            preferences[4] * No. of double visits *  1 +
#            preferences[5] * No. of single visits * -1

#F_weight : controller parameters
#F_weight        = np.array([ 1, 1, 1, 1, 1, 1, 1])  # all one
#F_weight        = np.array([2, 1, 1, 5, 3, 1, 2])  # educated guess
#F_weight        = np.array([ 2.90846782,  2.15963323,  9.48473502,  7.74506438,  4.69452669,  5.33303562, 9.55935917])    # learning result
F_weight        = [ 1.29964032,  9.83017599,  5.21240644,  6.3694487,   0.15822261,  7.11310888, 8.74563025]               # learning result

# F1: slew time cost 0~2
# F2: night urgency -1~1
# F3: overall urgency 0~1
# F4: altitude cost 0~1
# F5: hour angle cost 0~1
# F6: co-added depth cost 0~1
# F7: normalized brightness 0~1

s = time.time()

n_nights = 1 # number of the nights to be scheduled starting from 1st Sep. 2016



Date_start = ephem.Date('2020/12/31 12:00:00.00') # times are in UT


for i in range(n_nights):
    Date = Date_start + i # times are in UT

    # create scheduler
    scheduler = FBDE.Scheduler(Date, Site, F_weight)
    t2 = time.time()
    print('\nData imported in {} sec'.format(t2 - s))

    # schedule
    scheduler.schedule()
    t3 = time.time()
    print('\nScheduling finished in {} sec'.format(t3 - t2))

print('\n \nTotal elapsed time: {} sec'.format(time.time() - s))

