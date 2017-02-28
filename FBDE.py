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
        cur.execute('SELECT ID, Dec, RA, Label, N_visit, Last_visit, N_visit_u, Last_visit_u, N_visit_g, Last_visit_g, '
                    'N_visit_r, Last_visit_r, N_visit_i, Last_visit_i, N_visit_z, Last_visit_z, N_visit_y, Last_visit_y FROM FieldsStatistics')
        input1 = pd.DataFrame(cur.fetchall(), columns = ['ID', 'Dec', 'RA', 'Label', 'N_visit', 't_visit',
                                                         'N_visit_u', 't_visit_u', 'N_visit_r', 't_visit_r',
                                                         'N_visit_i', 't_visit_i', 'N_visit_g', 't_visit_g',
                                                         'N_visit_z', 't_visit_z', 'N_visit_y', 't_visit_y'])
        self.n_fields = len(input1)
        # create fields objects and feed their parameters and data
        dtype = [('ID', np.int), ('Dec', np.float), ('RA', np.float), ('Label', np.str), ('N_visit', np.int), ('t_visit', np.float),
                 ('N_visit_u', np.int), ('t_visit_u', np.float), ('N_visit_g', np.int), ('t_visit_g', np.float), ('N_visit_r', np.int),
                 ('t_visit_r', np.float), ('N_visit_i', np.int), ('t_visit_i', np.float), ('N_visit_z', np.int), ('t_visit_z', np.float),
                 ('N_visit_y', np.int), ('t_visit_y', np.float)]

        fields_info  = np.zeros((self.n_fields,), dtype = dtype)

        fields_info['ID']      = input1['ID']
        fields_info['Dec']     = input1['Dec']
        fields_info['RA']      = input1['RA']
        fields_info['Label']   = input1['Label']
        fields_info['N_visit'] = input1['N_visit']
        fields_info['t_visit'] = input1['t_visit']
        fields_info['N_visit_u'] = input1['N_visit_u']
        fields_info['t_visit_u'] = input1['t_visit_u']
        fields_info['N_visit_g'] = input1['N_visit_g']
        fields_info['t_visit_g'] = input1['t_visit_g']
        fields_info['N_visit_r'] = input1['N_visit_r']
        fields_info['t_visit_r'] = input1['t_visit_r']
        fields_info['N_visit_i'] = input1['N_visit_i']
        fields_info['t_visit_i'] = input1['t_visit_i']
        fields_info['N_visit_z'] = input1['N_visit_z']
        fields_info['t_visit_z'] = input1['t_visit_z']
        fields_info['N_visit_y'] = input1['N_visit_y']
        fields_info['t_visit_y'] = input1['t_visit_y']

        Max_N_visit = np.max(input1['N_visit']); Max_N_visit_u = np.max(input1['N_visit_u'])
        Max_N_visit_g = np.max(input1['N_visit_g']); Max_N_visit_r = np.max(input1['N_visit_r'])
        Max_N_visit_i = np.max(input1['N_visit_i']); Max_N_visit_z = np.max(input1['N_visit_z'])
        Max_N_visit_y = np.max(input1['N_visit_y'])


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
        # model param and
        cur.execute('SELECT Value FROM ModelParam')
        input3           = pd.DataFrame(cur.fetchall(), columns= ['ModelParam'])
        self.inf         = input3['ModelParam'][0]
        self.eps         = input3['ModelParam'][1]
        self.t_expo      = input3['ModelParam'][2]
        visit_w1         = input3['ModelParam'][3]; visit_w2 = input3['ModelParam'][4]
        self.visit_w     = [visit_w1, visit_w2]
        self.max_n_night = input3['ModelParam'][5]
        self.t_interval  = input3['ModelParam'][6]

        ''' Night variables'''
        Night_var = np.array([Max_N_visit, Max_N_visit_u, Max_N_visit_g,
                              Max_N_visit_r, Max_N_visit_i, Max_N_visit_z, Max_N_visit_y],
                             dtype = [('Max_n_visit', np.int), ('Max_n_visit_u', np.int),
                                      ('Max_n_visit_g', np.int), ('Max_n_visit_r', np.int),
                                      ('Max_n_visit_i', np.int), ('Max_n_visit_z', np.int), ('Max_n_visit_y', np.int)])



        self.fields = []
        for index, info in enumerate(fields_info):
            temp = FiledState(info, self.t_start, self.time_slots, all_fields_all_moments[index,:], slew_t[index,:],input3, Night_var)
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
        # scheduler decisions
        self.next_field   = None
        self.next_filter  = 'u'
        self.filter_change= None

    def schedule(self):
        self.episode.init_episode(self.fields)  # Initialize scheduling
        self.episode.field.update_visit_var(self.t_start, self.episode.last_filter)
        self.reset_output()

        while self.episode.t < self.episode.t_end:
            all_costs = np.zeros((self.n_fields,), dtype = [('u', np.float),('r', np.float),('i', np.float),('g', np.float),('z', np.float),('y', np.float)])
            for index, field in enumerate(self.fields):
                field.eval_feasibility()
                all_costs[index] = field.eval_cost(self.f_weight, self.next_filter)
            winner_indx, winner_cost, winner_filter = decision_maker(all_costs)

            # decisions made for this visit
            self.next_field    = self.fields[winner_indx]
            self.filter_change = (self.next_filter != None and self.next_filter != winner_filter)
            self.next_filter   = winner_filter

            # update visit variables of the next field
            t_visit = eval_t_visit(self.episode.t, self.next_field.slew_t_to, self.filter_change, 2 * ephem.minute)
            self.next_field.update_visit_var(t_visit, self.next_filter)
            # record visit
            self.record_visit()

            '''prepare for the next visit'''
            # update the episode status
            dt = eval_dt(self.next_field.slew_t_to, self.t_expo, self.filter_change, 2 * ephem.minute)
            self.episode.update_episode_var(dt, self.next_field, winner_filter)
            # update all fields
            self.episode.set_fields(self.fields, self.next_field)
        self.wrap_up()

    def reset_output(self):
        self.output_dtype = format_output()

        self.NightOutput  = np.zeros((0,), dtype =  self.output_dtype)
        try:
            os.remove("Output/log{}.lis".format(self.night_id))
        except:
            pass
        self.op_log = open("Output/log{}.lis".format(self.night_id),"w")

        #record the first entry
        entry1 = record_assistant(self.episode.field, self.episode.t, self.episode.last_filter, self.output_dtype, first_entry=True)
        self.NightOutput = np.append(self.NightOutput, entry1)
        self.op_log.write(json.dumps(entry1.tolist())+"\n")

    def record_visit(self):
        entry = record_assistant(self.next_field, self.episode.t, self.episode.last_filter, self.output_dtype)
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
        self.n          = None                 # current time slot
        self.step       = None                 # current decision number
        self.epi_prog   = None                 # Episode progress
        self.field      = None                 # current field

        self.third_last_filter  = None
        self.second_last_filter = None
        self.last_filter        = None
        self.n_f_change         = None                 # Number of filter change


    def init_episode(self, fields):
        self.clock(0, reset = True)
        self.set_filter(None, initialization = True)
        self.set_fields(fields, self.field, initialization = True)

    def update_episode_var(self, dt, field, filter):
        self.clock(dt)
        self.field = field
        self.set_filter(filter)

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


    def set_fields(self, fields, current_field, initialization = False):
        if initialization:
            self.field    = eval_init_state(fields, 0)
            current_field = self.field
        #finding the index of current field
        index = fields.index(current_field)
        for field in fields:
            field.update_field(self.n, self.t, index, initialization)

    def set_filter(self, visit_filter, initialization = False):
        if initialization:
            self.third_last_filter  = None
            self.second_last_filter = None
            self.last_filter        = eval_init_filter()
            self.n_f_change = 0
        else:
            if self.last_filter != visit_filter:
                self.n_f_change += 1
            self.third_last_filter  = self.second_last_filter
            self.second_last_filter = self.last_filter
            self.last_filter        = visit_filter




########################################################################################################################
########################################################################################################################

class FiledState(object): # an object of this class stores the information and status of a single field
    def __init__(self, field_info, t_start, time_slots, all_moments_data, all_slew_to ,model_param, Night_var):
        # parameters (constant during the current episode)
        # by input data
        self.id       = field_info['ID']
        self.dec      = field_info['Dec']
        self.ra       = field_info['RA']
        self.label    = field_info['Label']

        self.filter_dtype_count = [('all', np.int),('u', np.int),('g', np.int),('r', np.int),('i', np.int),('z', np.int),('y', np.int)]
        self.filter_dtype_value = [('all', np.float),('u', np.float),('g', np.float),('r', np.float),('i', np.float),('z', np.float),('y', np.float)]
        self.N_visit    = np.zeros(1, self.filter_dtype_count)
        self.t_visit    = np.zeros(1, self.filter_dtype_value)
        self.Max_N_visit= np.zeros(1, self.filter_dtype_count)

        self.N_visit['all']= field_info['N_visit'] # before the current episode of the scheduling
        self.t_visit['all']= field_info['t_visit'] # before the current episode of the scheduling
        self.N_visit['u']  = field_info['N_visit_u']
        self.t_visit['u']  = field_info['t_visit_u']
        self.N_visit['g']  = field_info['N_visit_g']
        self.t_visit['g']  = field_info['t_visit_g']
        self.N_visit['r']  = field_info['N_visit_r']
        self.t_visit['r']  = field_info['t_visit_r']
        self.N_visit['i']  = field_info['N_visit_i']
        self.t_visit['i']  = field_info['t_visit_i']
        self.N_visit['z']  = field_info['N_visit_z']
        self.t_visit['z']  = field_info['t_visit_z']
        self.N_visit['y']  = field_info['N_visit_y']
        self.t_visit['y']  = field_info['t_visit_y']

        self.Max_N_visit['all']  = Night_var[0]['Max_n_visit']
        self.Max_N_visit['u'] = Night_var[0]['Max_n_visit_u']
        self.Max_N_visit['g'] = Night_var[0]['Max_n_visit_g']
        self.Max_N_visit['r'] = Night_var[0]['Max_n_visit_r']
        self.Max_N_visit['i'] = Night_var[0]['Max_n_visit_i']
        self.Max_N_visit['z'] = Night_var[0]['Max_n_visit_z']
        self.Max_N_visit['y'] = Night_var[0]['Max_n_visit_y']

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
            self.n_ton_visits = np.zeros(1, self.filter_dtype_count)
            self.t_last_visit = np.zeros(1,self.filter_dtype_value)
            for index in range(7):
                self.t_last_visit[0][index] = -self.inf
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

    def update_visit_var(self, t_new_visit, filter):
        self.n_ton_visits[0]['all'] += 1
        self.t_last_visit[0]['all'] = t_new_visit

        if filter == 'u':
            self.n_ton_visits[0]['u'] += 1
            self.t_last_visit[0]['u'] = t_new_visit
        if filter == 'r':
            self.n_ton_visits[0]['r'] += 1
            self.t_last_visit[0]['r'] = t_new_visit
        if filter == 'i':
            self.n_ton_visits[0]['i'] += 1
            self.t_last_visit[0]['i'] = t_new_visit
        if filter == 'g':
            self.n_ton_visits[0]['g'] += 1
            self.t_last_visit[0]['g'] = t_new_visit
        if filter == 'z':
            self.n_ton_visits[0]['z'] += 1
            self.t_last_visit[0]['z'] = t_new_visit
        if filter == 'y':
            self.n_ton_visits[0]['y'] += 1
            self.t_last_visit[0]['y'] = t_new_visit

    def cal_param(self, t_start, time_slots):
        self.since_t_visit = np.zeros(1,self.filter_dtype_value)
        for index in range(7):
            if self.t_visit[0][index] == -self.inf:
                self.since_t_visit[0][index] = self.inf
            else:
                self.since_t_visit[0][index] = t_start - self.t_visit[0][index]

        r = np.where(self.all_moments_data['visible'])
        if (r[0].size):
            index = r[0][-1]
            self.t_setting = self.time_slots[index]
        else:
            self.t_setting = -self.inf

    def cal_variable(self, t):
        self.since_t_last_visit = np.zeros(1, self.filter_dtype_value)
        for index in self.since_t_last_visit.dtype.names:
            if self.t_last_visit[0][index] == -self.inf:
                self.since_t_last_visit[0][index] = self.inf
            else:
                self.since_t_last_visit[0][index] = t - self.t_last_visit[0][index]

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

    def eval_cost(self, f_weight, curr_filter):
        if not self.feasible:
            self.F = None
            return self.inf * np.ones((6,))
        self.F = eval_basis_fcn(self, curr_filter)
        self.cost = eval_cost(self.F, f_weight)
        return self.cost
