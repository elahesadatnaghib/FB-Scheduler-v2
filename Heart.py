__author__ = 'Elahe'


import ephem
import numpy as np
import json
from operator import attrgetter
from numpy import *
import time

''' Constants '''
inf = 1e10
eps = 1e-10


class Data(object):
    def __init__(self, date, site):
        self.Date   = date
        self.Site   = site

        ''' Predictable data '''
        # 3 by n_fields matrix of ID, RA, Dec
        self.all_fields = np.loadtxt("NightDataInLIS/Constants/fieldID.lis", dtype = "i4, f8, f8", unpack = True)
        self.time_slots = np.loadtxt("NightDataInLIS/TimeSlots{}.lis".format(int(ephem.julian_date(self.Date))), unpack = True)
        self.altitudes  = np.loadtxt("NightDataInLIS/Altitudes{}.lis".format(int(ephem.julian_date(self.Date))), unpack = True)
        self.hour_angs  = np.loadtxt("NightDataInLIS/HourAngs{}.lis".format(int(ephem.julian_date(self.Date))), unpack = True)
        #self.Moon_seps = np.loadtxt("MoonSeps{}.lis".format(int(ephem.julian_date(self.Date))), unpack = True)
        self.amass_cstr = np.loadtxt("NightDataInLIS/AirmassConstraints{}.lis".format(int(ephem.julian_date(self.Date))), unpack = True)
        self.all_n_tot_visits = np.loadtxt("NightDataInLIS/tot_N_visit{}.lis".format(int(ephem.julian_date(self.Date))), dtype = "i4", unpack = True)
        self.t_last_v_last= np.loadtxt("NightDataInLIS/t_last_visit{}.lis".format(int(ephem.julian_date(self.Date))), unpack = True)
        self.coad_depth   = self.all_n_tot_visits / (np.max(self.all_n_tot_visits) +1 ) #!!!!! temporarily!!!!!!!! # TODO Add coadded depth module instead of visit count
        self.vis_of_year  = np.zeros(len(self.all_fields[0]))   #!!!!! temporarily!!!!!!!! # TODO Visibility of the year is currently all zero
        self.sci_prog     = np.zeros(len(self.all_fields[0]), dtype= 'int')   #!!!!! temporarily!!!!!!!! # TODO Science program is not considered yet
        self.moon_sep     = np.loadtxt("NightDataInLIS/MoonSeps{}.lis".format(int(ephem.julian_date(self.Date))), unpack = True)
        # n_fields by n_fields symmetric matrix, slew time from field i to j
        self.slew_t     = np.loadtxt("NightDataInLIS/Constants/slewMatrix.dat", unpack = True) * ephem.second

        self.n_all_fields = len(self.all_fields[0])
        self.n_time_slots = len(self.time_slots)
        self.t_start      = self.time_slots[0]
        self.t_end        = self.time_slots[self.n_time_slots -1]
        self.n_start      = find_n(self.t_start, self.t_start, self.t_end, self.n_time_slots, self.time_slots)

        ''' Unpredictable data '''
        self.sky_brightness = np.zeros(len(self.all_fields[0]), dtype= 'int')   #!!!!! temporarily!!!!!!!!    # current sky brightness
        self.temp_coverage  = np.zeros(len(self.all_fields[0]), dtype= 'int')   #!!!!! temporarily!!!!!!!!    # temporary 0/1 coverage of the sky including clouds
        # TODO Add update module for live sky brightness and temporary coverage updates
        #print('\nData imported correctly') # data validity check should be added


    def update_sky_brightness(self, sky_brightness):     # SkyB is a 1 by n_fields vector, reflects the sky brightness at each field
        self.sky_brightness = sky_brightness            #must be fed into the algorithm in real time or as prediction in training

    def update_temp_coverage(self, temp_coverage):   # SkyB is a 1 by n_fields vector, reflects the sky brightness at each field
        self.temp_coverage = temp_coverage          #must be fed into the algorithm in real time or as prediction in training


class Scheduler(Data):
    def __init__(self, date, site, f_weight, preferences, manual_init_state = 0, exposure_t = 30 * ephem.second, visit_window = [15*ephem.minute, 30*ephem.minute], max_n_ton_visits =3, micro_train = False):
        super(Scheduler, self).__init__(date, site)

        # create telescope
        self.tonight_telescope = TelescopeState()
        self.tonight_telescope.set_param(self.t_start, self.t_end)

        # create fields objects and their parameters
        self.fields = []
        for index, field in enumerate(np.transpose(self.all_fields)):
            id   = field[0]
            ra   = field[1]
            dec  = field[2]
            temp = FiledState(id, ra, dec)
            t_rise = self.amass_cstr[0, index]
            set_t  = self.amass_cstr[1, index]
            temp.set_param(t_rise,
                           set_t,
                           self.all_n_tot_visits[index],
                           self.coad_depth[index],
                           self.vis_of_year[index],
                           self.sci_prog[index],
                           self.t_last_v_last[index])
            self.fields.append(temp)

        # scheduler outputs
        self.__NightOutput  = None
        self.__NightSummary = None

        # scheduler parameters
        self.exposure_t = exposure_t
        self.manual_init_state = manual_init_state
        self.visit_window = visit_window
        self.max_n_ton_visits = max_n_ton_visits
        self.f_weight     = f_weight
        self.preferences  = preferences

        # timing
        self.__t = None
        self.__n = None
        self.__step = None

        #other
        self.init_id = None

        # create trainer
        self.trainer = Trainer()
        self.micro_train = micro_train

    def set_f_wight(self, new_f_weight):
        self.f_weight = new_f_weight
    def get_f_wight(self):
        return self.f_weight

    def schedule(self):
        self.init_night()  #Initialize observation
        while self.__t < self.t_end:
            feasibility_idx = []
            all_costs       = np.ones(self.n_all_fields) * inf
            for field, index in zip(self.fields, range(self.n_all_fields)):
                if  self.is_feasible(field): # update features of the feasible fields
                    feasibility_idx.append(index)
                    self.update_field(field)
                    all_costs[index] = calculate_cost(field, self.tonight_telescope, self.f_weight)
            next_field_index, self.minimum_cost = decision_fcn(all_costs, feasibility_idx)
            self.next_field = self.fields[next_field_index]
            dt = self.next_field.slew_t_to + self.exposure_t
            if len(feasibility_idx) <= 10:
                print(len(feasibility_idx))
            self.clock(dt)
            # update next field visit variables
            self.next_field.update_visit_var(self.__t)
            self.tonight_telescope.update(self.__t, self.__n, self.__step, self.next_field, 0) # TODO Filter change decision making procedure (maybe as second stage decision)
            self.tonight_telescope.watch_fcn()
            self.record_visit()

            # update F_weights by feedback
            if self.micro_train:
                self.old_c_r = self.new_c_r
                self.new_c_r = self.cum_reward()
                reward = self.new_c_r - self.old_c_r - self.avg_rwd
                if reward > 0:
                    reward = 1
                elif reward < 0:
                    reward = -1
                self.avg_rwd = (self.avg_rwd * (self.__step -1) + (reward)) / float(self.__step)
                self.old_cost = self.new_cost
                self.new_cost = self.minimum_cost
                F_state= AllF[self.tonight_telescope.state.id -1]
                f_weight_correction = self.trainer.micro_feedback(R = reward, av_R = self.avg_rwd, n_C = self.new_cost, o_C = self.old_cost, F = F_state)
                self.set_f_wight(self.f_weight - f_weight_correction)
                self.AllF_weight = np.vstack((self.AllF_weight, self.f_weight))


        self.record_night()

    def update_field(self,field):
        id = field.id
        slew_t_to = self.calculate_f1(id)
        ha    = self.calculate_f4(id)
        t_to_invis = self.calculate_f6(field.set_t)
        normalized_bri = self.calculate_f7(id)
        cov = self.calculate_f10(id)
        field.set_soft_var(slew_t_to, ha, t_to_invis, normalized_bri, cov)


    def clock(self, dt, reset = False):
        if reset:
            self.__t = self.t_start + self.exposure_t
            self.__step = 0
        else:
            self.__t += dt
            self.__step += 1
        self.__n = find_n(self.__t, self.t_start, self. t_end, self.n_time_slots, self.time_slots)

    def init_night(self):
        # Reset Nights outputs
        self.__NightOutput  = np.zeros((1200,), dtype = [('Field_id', np.int),
                                                         ('ephemDate', np.float),
                                                         ('Filter', np.int),
                                                         ('n_ton', np.int),
                                                         ('n_last', np.int),
                                                         ('Cost', np.float),
                                                         ('Slew_t', np.float),
                                                         ('t_since_v_ton', np.float),
                                                         ('t_since_v_last', np.float),
                                                         ('Alt', np.float),
                                                         ('HA', np.float),
                                                         ('t_to_invis', np.float),
                                                         ('Sky_bri', np.float),
                                                         ('Temp_coverage', np.int)]) # at most 1200 visits per night
        self.__NightSummary = np.zeros(3) # t_start and t_end for now
        # Reset time
        self.clock(0,True)
        # Reset fields' state
        self.reset_fields_state()
        # Reset telescope
        init_state = self.init_state(self.manual_init_state, False)
        init_filter = self.init_filter()
        init_state.update_visit_var(self.__t)
        self.tonight_telescope.update(self.t_start, self.n_start, self.__step, init_state, init_filter)
        self.minimum_cost = 0.
        self.reset_feedback()
        # Record initial condition
        self.op_log = open("Output/log{}.lis".format(int(ephem.julian_date(self.Date))),"w")
        self.record_visit()

    def reset_fields_state(self):
        for index, field in enumerate(self.fields):
            alt = self.altitudes[0, index]
            ha  = self.hour_angs[0, index]
            cov = self.temp_coverage[index]
            bri = self.sky_brightness[index]
            t_last_visit = inf
            t_last_v_last = self.t_last_v_last[index]
            set_t         = self.amass_cstr[1, index]
            t_to_invis = self.calculate_f6(set_t)
            t_since_last_v_ton, t_since_last_v_last = self.calculate_f2(t_last_visit, t_last_v_last)
            slew_t_to = 0
            field.set_variables(alt, ha, cov, bri, t_to_invis, t_since_last_v_ton, t_since_last_v_last, slew_t_to)
            field.set_visit_var(0, t_last_visit)

    def init_state(self, state, manual = False):        # TODO Feasibility of the initial field needs to be checked
        if manual:
            self.init_id = state.id
            return state
        else:
            init_state = max(self.fields, key = attrgetter('alt'))
            self.init_id = init_state.id
            return init_state

    def init_filter(self):
        return 0

    def reset_feedback(self):
        self.old_c_r = 0
        self.new_c_r = 0
        self.AllF_weight = self.f_weight
        self.old_cost = 0
        self.new_cost = 0
        self.avg_rwd  = 0

    # Feature calculation
    def calculate_f1(self, id):     # slew time
        return self.slew_t[int(id -1), int(self.tonight_telescope.state.id) -1]

    def calculate_f2(self, t_last_v, t_last_v_last):# time since last visit
        if t_last_v_last != inf:
            t_since_last_v_last = self.__t - t_last_v_last
        else:
            t_since_last_v_last = inf
        if t_last_v != inf:
            t_since_last_v_ton = self.__t - t_last_v
        else:
            t_since_last_v_ton = inf
        return t_since_last_v_ton, t_since_last_v_last

    def calculate_f3(self, id):     # altitude
        return self.altitudes[self.__n, int(id) -1]

    def calculate_f4(self, id):     # hour angle
        return self.hour_angs[self.__n, int(id) -1]

    def calculate_f6(self, set_t):     # time to become effectively invisible- temporarily until setting below airmass horizon
        if set_t == 0:
            return inf
        else:
            return set_t - self.__t

    def calculate_f7(self, id):     # normalized sky brightness
        moon_size = 0.5 - np.abs(self.tonight_telescope.moon_phase - 0.5)
        moon_sep = self.moon_sep[self.__n, int(id) -1] / np.pi
        if moon_sep < 10 * np.pi/180:
            return inf
        if moon_size <0.2:
            return np.exp(-10 * moon_sep)
        elif moon_size < 0.5:
            return np.exp(-2 * moon_sep)
        elif moon_size < 0.8:
            return np.exp(-1 * moon_sep)
        else:
            return 1 - 0.5 * moon_sep

    def calculate_f8(self, id):     # visibility for rest of the year
        return 0
    def calculate_f9(self, id):     # science program identifier
        return 0
    def calculate_f10(self, id):    # 0/1 temporary coverage
        return 0



    def is_feasible(self, any_next_state):
        rise_t         = any_next_state.rise_t
        n_ton_visits = any_next_state.n_ton_visits
        t_last_v_last = any_next_state.t_last_v_last
        t_last_visit  = any_next_state.t_last_visit
        t_since_last_v_ton, t_since_last_v_last = self.calculate_f2(t_last_visit, t_last_v_last)
        current_field  = self.tonight_telescope.state.id
        id = any_next_state.id
        alt            = self.calculate_f3(id)
        slew_t         = self.calculate_f1(id)  #TODO change the slew_t and bri features from soft to hard variables
        bri            = self.calculate_f7(id)
        if rise_t != 0 and (any_next_state.rise_t > self.__t or any_next_state.set_t < self.__t):
            return False
        if rise_t == 0 and alt < np.pi/4:
            return False
        if t_since_last_v_ton != inf and (t_since_last_v_ton < self.visit_window[0] or t_since_last_v_ton > self.visit_window[1]):
            return False
        if n_ton_visits >= self.max_n_ton_visits:
            return False
        if current_field == any_next_state.id:
            return False
        if slew_t > 20 *ephem.second and t_since_last_v_ton != inf:
            return False
        if bri == inf:
            return False
        any_next_state.set_hard_var(t_since_last_v_ton, t_since_last_v_last, alt)
        return True

    def record_visit(self):
        self.__NightOutput[self.__step]['Field_id'] = self.tonight_telescope.state.id
        self.__NightOutput[self.__step]['ephemDate'] = self.__t
        self.__NightOutput[self.__step]['Filter'] = self.tonight_telescope.the_filter
        self.__NightOutput[self.__step]['n_ton'] = self.tonight_telescope.state.n_ton_visits
        self.__NightOutput[self.__step]['n_last'] = self.tonight_telescope.state.n_tot_visits
        self.__NightOutput[self.__step]['Cost'] = self.minimum_cost
        self.__NightOutput[self.__step]['Slew_t'] = self.tonight_telescope.state.slew_t_to
        self.__NightOutput[self.__step]['t_since_v_ton'] = self.tonight_telescope.state.t_since_last_v_ton
        self.__NightOutput[self.__step]['t_since_v_last'] = self.tonight_telescope.state.t_since_last_v_last
        self.__NightOutput[self.__step]['Alt'] = self.tonight_telescope.state.alt
        self.__NightOutput[self.__step]['HA'] = self.tonight_telescope.state.ha
        self.__NightOutput[self.__step]['t_to_invis'] = self.tonight_telescope.state.t_to_invis
        self.__NightOutput[self.__step]['Sky_bri'] = self.tonight_telescope.state.normalized_bri
        self.__NightOutput[self.__step]['Temp_coverage']= self.tonight_telescope.state.cov
        self.op_log.write(json.dumps(self.__NightOutput[self.__step].tolist())+"\n")

    def record_night(self):
        self.__NightSummary[0] = self.t_start
        self.__NightSummary[1] = self.t_end
        self.__NightSummary[2] = self.init_id
        np.save("Output/Schedule{}.npy".format(int(ephem.julian_date(self.Date))), self.__NightOutput)
        np.save("Output/Summary{}.npy".format(int(ephem.julian_date(self.Date))), self.__NightSummary)
        np.save("Output/Watch{}.npy".format(int(ephem.julian_date(self.Date))), self.tonight_telescope.watch)

    def performance(self):
        duration = (self.t_end - self.t_start) /ephem.hour
        # linear
        cost_avg = np.average(self.__NightOutput[0:self.__step]['Alt'])
        slew_avg = np.average(self.__NightOutput[0:self.__step]['Slew_t'])
        alt_avg  = np.average(self.__NightOutput[0:self.__step]['Alt'])
        # non-linear
        u, c           = np.unique(self.__NightOutput['Field_id'], return_counts=True)
        unique, counts = np.unique(c, return_counts=True)
        try:
            N_triple    = counts[unique == 3][0] / duration # per hour
        except:
            N_triple    = 0
        try:
            N_double    = counts[unique == 2][0] / duration
        except:
            N_double    = 0
        try:
            N_single    = counts[unique == 1][0] / duration
        except:
            N_single    = 0
        # objective function
        p = self.preferences[0] * cost_avg * -1 +\
            self.preferences[1] * slew_avg * -1 +\
            self.preferences[2] * alt_avg  *  1 +\
            self.preferences[3] * N_triple *  1 +\
            self.preferences[4] * N_double *  1 +\
            self.preferences[5] * N_single * -1

        return p

    def cum_reward(self):

        cost_sum = 0 #np.sum(self.__NightOutput[0:self.__step]['Alt'])
        slew_sum = 0 #np.sum(self.__NightOutput[0:self.__step]['Slew_t'])
        alt_sum  = 0 #np.sum(self.__NightOutput[0:self.__step]['Alt'])
        # non-linear
        u, c           = np.unique(self.__NightOutput['Field_id'], return_counts=True)
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

        # cumulative reward
        c_r =   self.preferences[0] * cost_sum * -1 +\
                self.preferences[1] * slew_sum * -1 +\
                self.preferences[2] * alt_sum  *  1 +\
                self.preferences[3] * N_triple *  1 +\
                self.preferences[4] * N_double *  1 +\
                self.preferences[5] * N_single * -1

        return c_r



class TelescopeState(object):
    def __init__(self):

        # variables
        self.t      = None                 # current time
        self.n      = None                 # current time slot
        self.__step = None                 # current decision number
        self.state  = None                 # current field
        self.the_filter = None


        # parameters
        self.t_start = None
        self.t_end   = None

        # Moon
        self.moon_phase = None

        # temporary
        self.watch = np.zeros((1200,), dtype = [('Field_id', np.int),
                                                ('ephemDate', np.float),
                                                ('F1', np.float),
                                                ('F2', np.float),
                                                ('F3', np.float),
                                                ('F4', np.float),
                                                ('F5', np.float),
                                                ('F6', np.float),
                                                ('F7', np.float)]) # at most 1200 visits per night

    def set_param(self, t_start, t_end):
        self.t_start = t_start
        self.t_end   = t_end
        self.moon_phase = (t_start - ephem.previous_new_moon(t_start))/30

    def set_t_n(self, t, n, step):
        self.t    = t
        self.n    = n
        self.step = step

    def set_state(self, state):
        self.state = state

    def set_filter(self,the_filter):
        self.the_filter = the_filter

    def update(self, t, n, step, state, the_filter):
        self.set_t_n(t, n, step)
        self.set_state(state)
        self.set_filter(the_filter)

    def watch_fcn(self, watch = True):
        if not watch:
            return
        else:
            F = AllF[int(self.state.id) -1]
            self.watch[self.step]['Field_id']  = self.state.id
            self.watch[self.step]['ephemDate'] = self.t
            self.watch[self.step]['F1']        = F[0]
            self.watch[self.step]['F2']        = F[1]
            self.watch[self.step]['F3']        = F[2]
            self.watch[self.step]['F4']        = F[3]
            self.watch[self.step]['F5']        = F[4]
            self.watch[self.step]['F6']        = F[5]
            self.watch[self.step]['F7']        = F[6]
            return



class FiledState(object):
    def __init__(self, id, ra, dec):
        # parameters (constant during the night)
        self.id  = id
        self.dec = dec
        self.ra  = ra
        self.rise_t = None
        self.set_t  = None
        self.n_tot_visits  = None # total number of visits before tonight
        self.coadded_depth = None # coadded depth before tonight
        self.vis_of_year   = None
        self.sci_prog      = None
        self.t_last_v_last = None

        # variables (gets updated after each time step)
        self.slew_t_to           = None
        self.t_since_last_v_ton  = None
        self.t_since_last_v_last = None
        self.alt                 = None
        self.ha                  = None
        self.t_to_invis          = None
        self.normalized_bri      = None
        self.cov                 = None

        # visit variables (gets updated only after a visit of itself)
        self.n_ton_visits = None # total number of tonight's visits
        self.t_last_visit = None # time of the last visit


    def set_param(self, rise_t, set_t, n_tot_visits, coad_depth, vis_of_year, sci_prog, t_last_v_last):
        self.rise_t        = rise_t
        self.set_t         = set_t
        self.n_tot_visits  = n_tot_visits
        self.coadded_depth = coad_depth
        self.vis_of_year   = vis_of_year
        self.sci_prog      = sci_prog
        self.t_last_v_last = t_last_v_last

    def set_variables(self, alt, ha, cov, bri, t_to_invis, t_since_last_v_ton, t_since_last_v_last, slew_t_to):
        self.slew_t_to = slew_t_to
        self.alt = alt
        self.ha = ha
        self.t_to_invis = t_to_invis
        self.normalized_bri = bri
        self.cov = cov
        self.t_since_last_v_ton = t_since_last_v_ton
        self.t_since_last_v_last = t_since_last_v_last

    def set_visit_var(self, n_ton_visits, t_new_visit):
        self.n_ton_visits = n_ton_visits
        self.t_last_visit = t_new_visit

    def update_visit_var(self,t_new_visit):
        self.n_ton_visits = self.n_ton_visits +1
        self.t_last_visit = t_new_visit

    # variables can be group as updated before feasibility check(hard) of after(soft)
    def set_hard_var(self, t_since_last_v, t_since_last_v_last, alt):
        self.t_since_last_v_ton  = t_since_last_v
        self.t_since_last_v_last = t_since_last_v_last
        self.alt                 = alt

    def set_soft_var(self, slew_t_to, ha, t_to_invis, normalized_bri, cov):
        self.slew_t_to      = slew_t_to
        self.ha             = ha
        self.t_to_invis     = t_to_invis
        self.normalized_bri = normalized_bri
        self.cov            = cov




# Basis function calculation

def calculate_F1(slew_t_to):            # slew time cost 0~2
    return (slew_t_to /ephem.second) /5

def calculate_F2(t_since_last_v_ton, n_ton_visits, t_to_invis):   # night urgency -1~1
    if t_since_last_v_ton == inf or n_ton_visits == 2:
        return 5
    elif n_ton_visits == 1:
        if t_to_invis < 30 * ephem.minute:
            return 0
        else:
            return 5 * (1 - np.exp(-1* t_since_last_v_ton / 20 * ephem.minute))

def calculate_F3(t_since_last_v_last):  # overall urgency 0~1
    if t_since_last_v_last == inf:
        return 0
    else:
        return 1/t_since_last_v_last

def calculate_F4(alt):                  # altitude cost 0~1
    return 1 - (2/np.pi) * alt

def calculate_F5(ha):                   # hour angle cost 0~1
    return np.abs(ha)/12

def calculate_F6(coadded_depth):        # coadded depth cost 0~1
    return coadded_depth

def calculate_F7(normalized_bri):       # normalized brightness 0~1
    return normalized_bri

# cost function

def cost_fcn(weight, F):
    return np.dot(weight, F)


def calculate_cost(possible_next_field, tonight_telescope, f_weight):
    slew_t_to          = possible_next_field.slew_t_to
    t_since_last_v_ton = possible_next_field.t_since_last_v_ton
    t_since_last_v_last= possible_next_field.t_since_last_v_last
    alt                = possible_next_field.alt
    ha                 = possible_next_field.ha
    n_ton_visits       = possible_next_field.n_ton_visits
    t_to_invis         = possible_next_field.t_to_invis
    coadded_depth      = possible_next_field.coadded_depth
    normalized_bri     = possible_next_field.normalized_bri
    F    = np.zeros(7)  # 7 is the number of basis functions
    F[0] = calculate_F1(slew_t_to)
    F[1] = calculate_F2(t_since_last_v_ton, n_ton_visits, t_to_invis)
    F[2] = calculate_F3(t_since_last_v_last)
    F[3] = calculate_F4(alt)
    F[4] = calculate_F5(ha)
    F[5] = calculate_F6(coadded_depth)
    F[6] = calculate_F7(normalized_bri)
    global AllF
    AllF[int(possible_next_field.id) -1] = F
    return cost_fcn(f_weight, F)


def decision_fcn(all_costs, feasibility_idx):
    cost_of_feasibles = [all_costs[i] for i in feasibility_idx]
    index = np.argmin(cost_of_feasibles)
    next_field_index = feasibility_idx[index]
    minimum_cost  = cost_of_feasibles[index]
    return next_field_index, minimum_cost

    # feasibility check and update some of the features that are used to check feasibility


# other functions
def find_n(t, t_start, t_end, n_time_slots, time_slots):
    n = 0
    if t <= t_start:
        return 0
    if t >= t_end:
        return  n_time_slots -1
    while t > time_slots[n]:
        n += 1
    return n



class Trainer(object):
        def micro_feedback(self, **options):
            old_cost   = options.get("o_C")
            new_cost   = options.get("n_C")
            reward    = options.get("R")
            F      = options.get("F")
            alpha  = 0.1
            gamma  = 0.1
            delta  = - old_cost - (reward - gamma * new_cost)
            F_w_correction = alpha * delta * F

            return F_w_correction




'''

Date            = ephem.Date('2016/09/01 12:00:00.00') # times are in UT
Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.

F_weight        = np.array([ 1, 1, 1, 1, 1, 1, 1])
# create scheduler
scheduler = Scheduler(Date, Site, F_weight)

# schedule
scheduler.schedule()
'''