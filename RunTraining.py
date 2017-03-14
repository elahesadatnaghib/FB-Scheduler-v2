__author__ = 'Elahe'


import numpy as np
import ephem
from BlackTraining import *
import time



Date            = ephem.Date('2015/6/28 12:00:00.00') +2 # times are in UT
Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.
F_weight        = [1, 1, 1, 1, 1, 1, 1]

preferences     = [1,1,4,0,3,10]
#P1: cost_av  * -1
#P2: slew_avg * -1
#P3: alt_avg  *  1
#P4: N_triple *  1
#P5: N_double *  1
#P6: N_single * -1

gray_train = False
black_train= True
custom_period = 0.1

s       = time.time()

if black_train:
    N_p     = 20
    F       = 0.8
    Cr      = 0.8
    maxIter = 20
    Domain  = np.array([[0,10], [0,10], [0,10], [0,10], [0,10], [0,10], [0,10]])
    D       = 7
    train   = BlackTraining(Date, Site, preferences, gray_train, custom_period)
    train.DE_opt(N_p, F, Cr, maxIter, D, Domain)

elif gray_train:
    scheduler = FBDE.Scheduler(Date, Site, F_weight, gray_train, custom_period)

print('Total elapsed time: {} sec'.format(time.time() - s))