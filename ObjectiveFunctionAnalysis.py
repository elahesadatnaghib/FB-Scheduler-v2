__author__ = 'Elahe'

import numpy as np
import ephem
import FBDE

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

Domain  = np.array([[0,10], [0,10], [0,10], [0,10], [0,10], [0,10], [0,10]])

# returns a uniform grid of sampling in the space of solutions
def uniform_sampleing_points(number_of_sampling, F_weight_sample, domain):




