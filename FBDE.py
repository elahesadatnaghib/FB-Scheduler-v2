__author__ = 'Elahe'


import ephem
import numpy as np
import json
from numpy import *
import sqlite3 as lite
from calculations import *
import pandas as pd


class DataFeed(object):
    def __init__(self, date, site, custom_episode = False):
        self.Site   = site

        self.night_id = int(date - ephem.Date('2020/12/31 12:00:00.00')) + 1

        # connecting to db
        con = lite.connect('FBDE.db')
        cur = con.cursor()

        # fields data: ID, RA, Dec, Label, N_visit, time of the last visit
        cur.execute('SELECT ID, Dec, RA, Label, N_visit, Last_visit FROM FieldsStatistics')
        input1 = pd.DataFrame(cur.fetchall(), columns = ['ID', 'Dec', 'RA', 'Label', 'N_visit', 't_visit'])
        self.n_fields = len(input1)
        # create fields objects and feed their parameters and data
        dtype = [('ID', np.int), ('Dec', np.float), ('RA', np.float), ('Label', np.str), ('N_visit', np.int), ('t_visit', np.float)]
        fields_info  = np.zeros((self.n_fields,), dtype = dtype)

        fields_info['ID']      = input1['ID']
        fields_info['Dec']     = input1['Dec']
        fields_info['RA']      = input1['RA']
        fields_info['Label']   = input1['Label']
        fields_info['N_visit'] = input1['N_visit']
        fields_info['t_visit'] = input1['t_visit']
        del input1

        ''' import data for the  current night '''
        cur.execute('SELECT ephemDate, altitude, hourangle, visible, covered, brightness FROM FieldData where nightid == {}'.format(self.night_id))
        input2 = pd.DataFrame(cur.fetchall(), columns=['ephemDate', 'alts','hourangs', 'visible', 'covered', 'brightness'])

        self.n_t_slots = (np.shape(input2)[0]) / self.n_fields
        all_fields_all_moments = np.zeros((self.n_fields,self.n_t_slots,), dtype =  [('alts', np.float),
                                                                                     ('hourangs', np.float),
                                                                                     ('visible', np.bool),
                                                                                     ('covered', np.bool),
                                                                                     ('brightness', np.float)])

        self.time_slots =  np.zeros(self.n_t_slots)
        self.time_slots = input2['ephemDate'][0:self.n_t_slots]

        for i in range(self.n_fields):
            all_fields_all_moments[i, :]['alts']       = input2['alts'][i * self.n_t_slots : (i+1) * self.n_t_slots]
            all_fields_all_moments[i, :]['hourangs']   = input2['hourangs'][i * self.n_t_slots : (i+1) * self.n_t_slots]
            all_fields_all_moments[i, :]['visible']    = input2['visible'][i * self.n_t_slots : (i+1) * self.n_t_slots]
            all_fields_all_moments[i, :]['covered']    = input2['covered'][i * self.n_t_slots : (i+1) * self.n_t_slots] #TODO covered and brighntess should be updatable
            all_fields_all_moments[i, :]['brightness'] = input2['brightness'][i * self.n_t_slots : (i+1) * self.n_t_slots]
        del input2


        # adjusting t start and t end to where the data exist
        self.t_start = self.time_slots[0]; self.t_end = self.time_slots[self.n_t_slots -1]
        # n_fields by n_fields symmetric matrix, slew time from field i to j
        slew_t = np.loadtxt("NightDataInLIS/Constants/slewMatrix.dat") * ephem.second

        ''' Model parameter and data'''
        # model param
        cur.execute('SELECT Value FROM ModelParam')
        input3           = pd.DataFrame(cur.fetchall(), columns= ['ModelParam'])
        self.inf         = input3['ModelParam'][0]
        self.eps         = input3['ModelParam'][1]
        self.t_expo      = input3['ModelParam'][2]
        visit_w1         = input3['ModelParam'][3]; visit_w2 = input3['ModelParam'][4]
        self.visit_w     = [visit_w1, visit_w2]
        self.max_n_night = input3['ModelParam'][5]
        self.t_interval  = input3['ModelParam'][6]


        self.fields = []
        for index, info in enumerate(fields_info):
            temp = FiledState(info, self.t_start, self.time_slots, all_fields_all_moments[index,:], slew_t[index,:],input3)
            self.fields.append(temp)

        del all_fields_all_moments
        del slew_t
        del input3
        con.close()
        # create episode
        self.episode = EpisodeStatus(self.t_start, self.t_end, self.time_slots, self.t_expo)

########################################################################################################################
########################################################################################################################

class Scheduler(DataFeed):
    def __init__(self, date, site, f_weight):
        super(Scheduler, self).__init__(date, site)

        # scheduler parameters
        self.f_weight     = f_weight
        self.next_field   = None

    def schedule(self):
        self.episode.init_episode(self.fields)  # Initialize scheduling
        self.episode.field.update_visit_var(self.t_start)
        self.reset_output()

        while self.episode.t < self.episode.t_end:
            all_costs = np.zeros(self.n_fields)
            for index, field in enumerate(self.fields):
                field.eval_feasibility()
                all_costs[index] = field.eval_cost(self.f_weight)
            winner_indx, min_cost = decision_maker(all_costs)
            self.next_field = self.fields[winner_indx]
            # update visit variables of the next field
            t_visit = eval_t_visit(self.episode.t, self.next_field.slew_t_to)
            self.next_field.update_visit_var(t_visit)
            # record visit
            self.record_visit()

            '''prepare for the next visit'''
            # update the episode status
            dt = eval_dt(self.next_field.slew_t_to, self.t_expo)
            self.episode.update_episode_var(dt, self.next_field, 'r')
            # update all fields
            self.episode.set_fields(self.fields, self.next_field)
        self.wrap_up()

    def reset_output(self):
        self.output_dtype = [('Field_id', np.int),
                             ('ephemDate', np.float),
                             ('Filter', np.str),
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
                             ('Temp_coverage', np.int),
                             ('F1', np.float),
                             ('F2', np.float),
                             ('F3', np.float),
                             ('F4', np.float),
                             ('F5', np.float),
                             ('F6', np.float),
                             ('F7', np.float)]

        self.NightOutput  = np.zeros((0,), dtype =  self.output_dtype)

        try:
            os.remove("Output/log{}.lis".format(self.night_id))
        except:
            pass
        self.op_log = open("Output/log{}.lis".format(self.night_id),"w")

        #record the first entry
        entry1 = record_assistant(self.episode.field, self.episode.t, self.episode.filter, self.output_dtype, first_entry=True)
        self.NightOutput = np.append(self.NightOutput, entry1)
        self.op_log.write(json.dumps(entry1.tolist())+"\n")

    def record_visit(self):
        entry = record_assistant(self.next_field, self.episode.t, self.episode.filter, self.output_dtype)
        self.NightOutput = np.append(self.NightOutput, entry)
        self.op_log.write(json.dumps(entry.tolist())+"\n")

    def wrap_up(self):
        np.save("Output/Schedule{}.npy".format(self.night_id), self.NightOutput)

    def set_f_wight(self, new_f_weight):
        self.f_weight = new_f_weight

    def eval_performance(self, preferences):
        return eval_performance(self.NightOutput, preferences)

########################################################################################################################
########################################################################################################################

class EpisodeStatus(object):
    def __init__(self, t_start, t_end, time_slots, exposure_t):
        # parameters constant during the current episode
        self.t_start    = t_start
        self.t_end      = t_end
        self.episode_len= t_end - t_start # in days
        self.time_slots = time_slots
        self.n_t_slots  = len(time_slots)
        self.exposure_t = exposure_t
        self.moon_phase = (t_start - ephem.previous_new_moon(t_start))/30 # need to be changed

        # variables change after each decision
        self.t          = None                 # current time
        self.n          = None                # current time slot
        self.step       = None                 # current decision number
        self.epi_prog   = None                 # Episode progress
        self.field      = None                 # current field
        self.filter     = None


    def init_episode(self, fields):
        self.reset_episode(fields)
        self.set_fields(fields, self.field, initialization = True)

    def update_episode_var(self, dt, field, filter):
        self.clock(dt)
        self.field = field
        self.filter= filter

    def clock(self, dt, reset = False): # sets or resets t, n, step
        if reset:
            self.t = self.t_start + self.exposure_t
            self.step = 0
        else:
            self.t += dt
            self.step += 1
        self.find_n()

    def find_n(self):
        n = 0
        if self.t <= self.t_start:
            self.n = 0
        elif self.t >= self.t_end:
            self.n = self.n_t_slots -1
        else:
            while self.t > self.time_slots[n]:
                n += 1
            self.n = n

    def reset_episode(self, fields):
        self.clock(0, reset = True)
        self.filter= eval_init_filter()
        self.field = eval_init_state(fields, 0)

    def set_fields(self, fields, current_field, initialization = False):
        #finding the index of current field
        index = fields.index(current_field)
        for field in fields:
            field.update_field(self.n, self.t, index, initialization)


########################################################################################################################
########################################################################################################################

class FiledState(object): # an object of this class stores the information and status of a single field
    def __init__(self, field_info, t_start, time_slots, all_moments_data, all_slew_to ,model_param):
        # parameters (constant during the current episode)
        # by input data
        self.id       = field_info['ID']
        self.dec      = field_info['Dec']
        self.ra       = field_info['RA']
        self.label    = field_info['Label']

        self.N_visit  = field_info['N_visit'] # before the current episode of the scheduling
        self.t_visit  = field_info['t_visit'] # before the current episode of the scheduling
        #self.year_vis =                      # visibility of the year to be added
        # by calculation
        self.time_slots    = time_slots
        self.since_t_visit = None
        self.t_setting     = None
        # parameters that based on the updates to prediction might need to be updated
        #self.night_vis=                      # prediction for the visibility of the night to be added (with cloud model)

        # variables (gets updated after each time step for all field)
        # from input data
        self.slew_t_to           = None
        self.alt                 = None
        self.ha                  = None
        self.visible             = None
        self.brightness          = None
        self.covered             = None
        # by calculation
        self.since_t_last_visit  = None
        self.t_to_invis          = None
        self.feasible            = None
        self.cost                = None

        # visit variables (gets updated only after a visit of the specific field)
        # from input data
        self.n_ton_visits = None # total number of visits in the current episode
        self.t_last_visit = None # time of the last visit in the current episode

        # data of the field for all moments of the current episode
        self.all_moments_data = None
        self.all_slew_to      = None

        #Basis functions
        self.F = None

        self.data_feed(all_moments_data, all_slew_to, model_param)
        self.cal_param(t_start, time_slots)

    def update_field(self, n, t, current_state_index, initialization = False):
        self.slew_t_to = self.all_slew_to[current_state_index]
        self.alt       = self.all_moments_data[n]['alts']
        self.ha        = self.all_moments_data[n]['hourangs']
        self.visible   = self.all_moments_data[n]['visible']
        self.brightness= self.all_moments_data[n]['brightness']
        self.covered   = self.all_moments_data[n]['covered']
        if initialization :
            self.n_ton_visits = 0
            self.t_last_visit = -self.inf
        # must be executed after all the variables are updated
        self.cal_variable(t)

    def set_param(self, night_visibility):
        self.night_vis = night_visibility

    def set_variable(self, slew_t_to, alt, ha, bri, cov, t):
        self.slew_t_to = slew_t_to
        self.alt       = alt
        self.ha        = ha
        self.brightness= bri
        self.covered   = cov
        self.cal_variable(t)

    def update_visit_var(self, t_new_visit):
        self.n_ton_visits += 1
        self.t_last_visit = t_new_visit

    def cal_param(self, t_start, time_slots):
        if self.t_visit == -self.inf:
            self.since_t_visit = self.inf
        else:
            self.since_t_visit = t_start - self.t_visit

        range = np.where(self.all_moments_data['visible'])
        if (range[0].size):
            index = range[0][-1]
            self.t_setting = self.time_slots[index]
        else:
            self.t_setting = -self.inf

    def cal_variable(self, t):
        if self.t_last_visit == -self.inf:
            self.since_t_last_visit = self.inf
        else:
            self.since_t_last_visit = t - self.t_last_visit

        if self.t_setting == -self.inf:
            self.t_to_invis = -self.inf
        else:
            self.t_to_invis = self.t_setting - t
            if self.t_to_invis < self.t_interval /2:
                self.t_to_invis = 0



    def data_feed(self, all_moments_data, all_slew_to, model_param):
        self.all_moments_data = all_moments_data
        self.all_slew_to     = all_slew_to
        self.inf         = model_param['ModelParam'][0]
        self.eps         = model_param['ModelParam'][1]
        self.t_expo      = model_param['ModelParam'][2]
        visit_w1         = model_param['ModelParam'][3]; visit_w2 = model_param['ModelParam'][4]
        self.visit_w     = [visit_w1, visit_w2]
        self.max_n_night = model_param['ModelParam'][5]
        self.t_interval  = model_param['ModelParam'][6]

    def eval_feasibility(self):
        self.feasible = eval_feasibility(self)
        return self.feasible

    def eval_cost(self, f_weight):
        if not self.feasible:
            self.F = None
            return self.inf
        self.F = eval_basis_fcn(self)
        self.cost = eval_cost(self.F, f_weight)
        return self.cost



'''

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




        self.altitudes  =  np.zeros([self.n_t_slots,self.n_fields])
        self.hour_angs  =  np.zeros([self.n_t_slots,self.n_fields])
        self.visible    =  np.zeros([self.n_t_slots,self.n_fields], dtype = bool)
        self.covered    =  np.zeros([self.n_t_slots,self.n_fields], dtype = bool)
        self.brightness =  np.zeros([self.n_t_slots,self.n_fields])
        self.time_slots =  np.zeros(self.n_t_slots)


        for i in range(self.n_t_slots):
            self.altitudes[i,] = input2[4][i * self.n_fields : (i+1) * self.n_fields]
            self.hour_angs[i,] = input2[6][i * self.n_fields : (i+1) * self.n_fields]
            self.visible[i,]   = input2[8][i * self.n_fields : (i+1) * self.n_fields]
            self.covered[i,]   = input2[9][i * self.n_fields : (i+1) * self.n_fields] #TODO covered and brighntess should be updatable
            self.brightness[i,]=input2[10][i * self.n_fields : (i+1) * self.n_fields]
            self.time_slots[i] = input2[2][i]
        del input2

        '''