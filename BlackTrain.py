__author__ = 'Elahe'


import numpy as np
import ephem
import FBDE
import MyDE
import time
import os




Date            = ephem.Date('2015/6/28 12:00:00.00')+9 # times are in UT
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

class Training():
    def __init__(self, Date, Site, preferences):
        Site.lon        = -1.2320792
        Site.lat        = -0.517781017
        Site.elevation  = 2650
        Site.pressure   = 0.
        Site.horizon    = 0.
        F_weight        = [1, 1, 1, 1, 1, 1, 1]
        self.scheduler  = FBDE.Scheduler(Date, Site, F_weight)
        self.pref       = preferences

    def DE_opt(self, N_p, F, Cr, maxIter, D, domain):
        self.D               = D
        self.domain          = domain
        self.optimizer       = MyDE.DE_optimizer(self, N_p, F, Cr, maxIter)

    def target(self, x):
        self.scheduler.set_f_wight(x)
        self.scheduler.schedule()
        return -1 * self.scheduler.eval_performance(self.pref)




s       = time.time()
train   = Training(Date, Site, preferences)

N_p     = 20
F       = 0.8
Cr      = 0.8
maxIter = 20
Domain  = np.array([[0,10], [0,10], [0,10], [0,10], [0,10], [0,10], [0,10]])
D       = 7
train.DE_opt(N_p, F, Cr, maxIter, D, Domain)

print('Total elapsed time: {} sec'.format(time.time() - s))