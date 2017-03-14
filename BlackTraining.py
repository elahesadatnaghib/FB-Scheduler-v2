__author__ = 'Elahe'


import MyDE
import FBDE


class BlackTraining():
    def __init__(self, Date, Site, preferences, gray_train = False, custom_period = 0):
        Site.lon        = -1.2320792
        Site.lat        = -0.517781017
        Site.elevation  = 2650
        Site.pressure   = 0.
        Site.horizon    = 0.
        F_weight        = [1, 1, 1, 1, 1, 1, 1]
        self.scheduler  = FBDE.Scheduler(Date, Site, F_weight, gray_train, custom_period)
        self.pref       = preferences

    def DE_opt(self, N_p, F, Cr, maxIter, D, domain):
        self.D               = D
        self.domain          = domain
        self.optimizer       = MyDE.DE_optimizer(self, N_p, F, Cr, maxIter)

    def target(self, x):
        self.scheduler.set_f_wight(x)
        self.scheduler.schedule()
        return -1 * self.scheduler.eval_performance(self.pref)


