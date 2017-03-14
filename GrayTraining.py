__author__ = 'Elahe'

import numpy as np
from calculations import *


class GrayTrainer(object):
    def __init__(self):
        self.update_period = 10
        self.learning_rate = 0.1

    def train(self, scheduler_out, f_weight, preferences):
        f_weight_cor = self.eval_new_f_weight(scheduler_out, preferences)
        new_f_weight = f_weight - f_weight_cor
        return new_f_weight

    def eval_new_f_weight(self, scheduler_out, preferences):
        G0 = eval_performance(scheduler_out[0:-self.update_period], preferences)
        G1 = eval_performance(scheduler_out, preferences)
        del_G = G1 - G0
        del_C = self.eval_del_C(scheduler_out)

        del_O = del_C - del_G
        sum_F = self.eval_sum_F(scheduler_out)

        del_f_weight = del_O / sum_F
        for i,del_f in enumerate(del_f_weight):
            if del_f > 0:
                del_f_weight[i] = 0.1
            if del_f < 0:
                del_f_weight[i] = -0.1

        print(G0,G1,del_G,del_C,del_f_weight)
        return del_f_weight * self.learning_rate

    def eval_del_C(self, scheduler_out):
        len_output = len(scheduler_out)
        #print(scheduler_out[len_output - self.update_period:]['Cost'])
        return np.sum(scheduler_out[len_output - self.update_period:]['Cost'])

    def eval_sum_F(self, scheduler_out):
        len_output = len(scheduler_out)
        output_seg =scheduler_out[len_output - self.update_period:]
        s_F1 = np.sum(output_seg['F1'])
        s_F2 = np.sum(output_seg['F2'])
        s_F3 = np.sum(output_seg['F3'])
        s_F4 = np.sum(output_seg['F4'])
        s_F5 = np.sum(output_seg['F5'])
        s_F6 = np.sum(output_seg['F6'])
        s_F7 = np.sum(output_seg['F7'])
        s_F = np.array([s_F1, s_F2, s_F3, s_F4, s_F5, s_F6, s_F7])
        return  s_F




class WatchData(object):
    def __init__(self, n_fields):
        self.n_fields = n_fields
        self.steps_tp_save        = [0,10,11,12,500]
        self.steps_tp_save_indx   = 0
        self.n_entries            = 8+48
        self.data_vec = np.zeros([self.n_fields +1, self.n_entries])
        self.data_vec_index = 0

    def watch_save_one_entry(self,field, step):
        if step == self.steps_tp_save[self.steps_tp_save_indx]:
            entry = np.array([field.id, field.N_visit[0]['all'], field.slew_t_to, field.alt, field.covered, field.moonsep,
                              field.n_ton_visits[0]['all'], field.since_t_last_visit[0]['all'], #8 elements
                              field.cost[0]['u'], field.cost[0]['g'], field.cost[0]['r'], field.cost[0]['i'], field.cost[0]['z'], field.cost[0]['y'],
                              field.F[0]['u'], field.F[0]['g'], field.F[0]['r'], field.F[0]['i'], field.F[0]['z'], field.F[0]['y'],
                              field.F[1]['u'], field.F[1]['g'], field.F[1]['r'], field.F[1]['i'], field.F[1]['z'], field.F[1]['y'],
                              field.F[2]['u'], field.F[2]['g'], field.F[2]['r'], field.F[2]['i'], field.F[2]['z'], field.F[2]['y'],
                              field.F[3]['u'], field.F[3]['g'], field.F[3]['r'], field.F[3]['i'], field.F[3]['z'], field.F[3]['y'],
                              field.F[4]['u'], field.F[4]['g'], field.F[4]['r'], field.F[4]['i'], field.F[4]['z'], field.F[4]['y'],
                              field.F[5]['u'], field.F[5]['g'], field.F[5]['r'], field.F[5]['i'], field.F[5]['z'], field.F[5]['y'],
                              field.F[6]['u'], field.F[6]['g'], field.F[6]['r'], field.F[6]['i'], field.F[6]['z'], field.F[6]['y']])
            self.data_vec[self.data_vec_index, :] = entry
            self.data_vec_index += 1

    def watch_last_entry(self,step,t, next_field, next_filter, filter_change, winning_cost):
        if step == self.steps_tp_save[self.steps_tp_save_indx]:
            entry = np.zeros([1,self.n_entries])
            entry[0,0] = step
            entry[0,1] = t
            entry[0,2] = next_field
            entry[0,3] = next_filter
            entry[0,4] = filter_change
            entry[0,5] = winning_cost
            self.data_vec[self.data_vec_index, :] = entry

    def save_watched_data(self, step):
        if step == self.steps_tp_save[self.steps_tp_save_indx]:
            np.save('Watch/Data{}.npy'.format(step), self.data_vec)
            if self.steps_tp_save_indx < len(self.steps_tp_save) -1:
                self.steps_tp_save_indx +=1
            print('Data{} is out'.format(step))
            self.data_vec = self.data_vec = np.zeros([self.n_fields +1, self.n_entries])
            self.data_vec_index = 0

