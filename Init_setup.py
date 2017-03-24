__author__ = 'Elahe'

# This is to
#1. creat the database
#2. generate the field data (indep of teh schedule)


import numpy as np
import ephem
import FBDE
import time

# my modules

import CreateDB
import FieldDataGenerator as dgen

def initial_setup(site, end_date):

    # create database and data structures
    CreateDB.creatFBDE()

    # generate field data
    dgen.Field_data(site, end_date)



# execution
s = time.time()

Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.

#Number of the nights to generate the data for
n_nights = 2

#end_date        = ephem.Date('2020/12/31 12:00:00.00') + n_nights
end_date        = ephem.Date('2015/6/28 12:00:00.00') + n_nights

initial_setup(Site, end_date)

print('Total elapsed time: {} minutes'.format((time.time() - s)/60))




