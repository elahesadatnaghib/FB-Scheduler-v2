__author__ = 'Elahe'

import numpy as np
import ephem
from operator import attrgetter


def eval_init_state(fields, suggestion, manual = False):        # TODO Feasibility of the initial field needs to be checked
    if manual:
        return suggestion
    else:
        #build a vector of all altitudes at t start
        all_alt_start = [field.all_moments_data[0]['alts'] for field in fields]
        winner_index = np.argmax(all_alt_start)
        return fields[winner_index]

def eval_init_filter():
    return 'r'

def eval_feasibility(field):
    if not field.visible:
        return False
    elif field.n_ton_visits[0]['all'] >= field.max_n_night:
        return False
    elif field.n_ton_visits[0]['all'] == 1 and field.since_t_last_visit[0]['all'] < field.visit_w[0]:
        return False
    elif field.n_ton_visits[0]['all'] == 1 and field.since_t_last_visit[0]['all'] > field.visit_w[1]:
        return False
    elif field.covered:
        return False
    if field.slew_t_to > 5 *ephem.second and field.since_t_last_visit[0]['all'] != field.inf:
        return False
    return True

def eval_basis_fcn(field):
    F    = np.zeros(7, dtype = [('u', np.float),('r', np.float),('i', np.float),('g', np.float),('z', np.float),('y', np.float)])  # 7 is the number of basis functions
    for index in F.dtype.names:
        F[0][index] = calculate_F1(field.slew_t_to)
        F[1][index] = calculate_F2(field.since_t_last_visit[0][index], field.n_ton_visits[0][index], field.t_to_invis, field.inf)
        F[2][index] = calculate_F3(field.since_t_visit[0][index], field.inf)
        F[3][index] = calculate_F4(field.alt)
        F[4][index] = calculate_F5(field.ha)
        F[5][index] = calculate_F6(field.N_visit[0][index])
        F[6][index] = calculate_F7(field.brightness)
    return F

def eval_cost(F, f_weight):
    c = np.zeros(1, dtype = [('u', np.float),('r', np.float),('i', np.float),('g', np.float),('z', np.float),('y', np.float)])
    for index in F.dtype.names:
        c[0][index] = np.dot(F[:][index],f_weight)
    return c

def decision_maker(all_costs):
    min_cost = np.zeros(1, dtype = [('u', np.float),('r', np.float),('i', np.float),('g', np.float),('z', np.float),('y', np.float)])
    min_index = np.zeros(1, dtype = [('u', np.int),('r', np.int),('i', np.int),('g', np.int),('z', np.int),('y', np.int)])

    winner_filter = 'u'
    for index in min_cost.dtype.names:
        min_index[index] = np.argmin(all_costs[:][index])
        min_cost[0][index]  = all_costs[min_index[index]][index]
        if min_cost[0][index] < min_cost[0][winner_filter]:
            winner_filter = index

    winner_index = min_index[0][winner_filter]
    winner_cost  =  min_cost[0][winner_filter]

    #TODO check for close competitors
    return winner_index, winner_cost, winner_filter

def eval_dt(slew_t_to, t_expo, filter_change, filter_change_t):
    if filter_change:
        return slew_t_to + t_expo + filter_change_t
    else:
        return slew_t_to + t_expo


def eval_t_visit(t_decision, slew_t_to, filter_change, filter_change_t):
    if filter_change:
        return t_decision + slew_t_to + filter_change_t
    else:
        return t_decision + slew_t_to

def eval_performance(episode_output, preferences):
    t_start = episode_output[0]['ephemDate']
    t_end   = episode_output[-1]['ephemDate']
    duration = (t_end - t_start) /ephem.hour

    # linear
    cost_avg = np.average(episode_output['Cost'])
    slew_avg = np.average(episode_output['Slew_t'])
    alt_avg  = np.average(episode_output['Alt'])
    # non-linear
    u, c           = np.unique(episode_output['Field_id'], return_counts=True)
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
    p = preferences[0] * cost_avg * -1 +\
        preferences[1] * slew_avg * -1 +\
        preferences[2] * alt_avg  *  1 +\
        preferences[3] * N_triple *  1 +\
        preferences[4] * N_double *  1 +\
        preferences[5] * N_single * -1

    return p


def calculate_F1(slew_t_to):            # slew time cost 0~2
    return (slew_t_to /ephem.second) /5

def calculate_F2(since_t_last_visit, n_ton_visits, t_to_invis, inf):   # night urgency -1~1
    if since_t_last_visit == inf or n_ton_visits == 2:
        return 5
    elif n_ton_visits == 1:
        if t_to_invis < 30 * ephem.minute:
            return 0
        else:
            return 5 * (1 - np.exp(-1* since_t_last_visit / 20 * ephem.minute))

def calculate_F3(since_t_visit, inf):  # overall urgency 0~1
    if since_t_visit == inf:
        return 0
    else:
        return 1/since_t_visit

def calculate_F4(alt):                  # altitude cost 0~1
    return 1 - (2/np.pi) * alt

def calculate_F5(ha):                   # hour angle cost 0~1
    return np.abs(ha)/12

def calculate_F6(N_visit):        # coadded depth cost 0~1
    return N_visit

def calculate_F7(brightness):       # normalized brightness 0~1
    return brightness




# miscellaneous
def record_assistant(field, t, filter, output_dtype, first_entry = False):
    if first_entry:
        entry = np.array((field.id,
                          float(t),
                          filter,
                          field.n_ton_visits[0]['all'],
                          field.N_visit[0]['all'],
                          field.n_ton_visits[0]['u'],
                          field.N_visit[0]['u'],
                          field.n_ton_visits[0]['g'],
                          field.N_visit[0]['g'],
                          field.n_ton_visits[0]['r'],
                          field.N_visit[0]['r'],
                          field.n_ton_visits[0]['i'],
                          field.N_visit[0]['i'],
                          field.n_ton_visits[0]['z'],
                          field.N_visit[0]['z'],
                          field.n_ton_visits[0]['y'],
                          field.N_visit[0]['y'],
                          0.,
                          0.,
                          field.since_t_last_visit[0]['all'],
                          field.since_t_visit[0]['all'],
                          field.since_t_last_visit[0]['u'],
                          field.since_t_visit[0]['u'],
                          field.since_t_last_visit[0]['g'],
                          field.since_t_visit[0]['g'],
                          field.since_t_last_visit[0]['r'],
                          field.since_t_visit[0]['r'],
                          field.since_t_last_visit[0]['i'],
                          field.since_t_visit[0]['i'],
                          field.since_t_last_visit[0]['z'],
                          field.since_t_visit[0]['z'],
                          field.since_t_last_visit[0]['y'],
                          field.since_t_visit[0]['y'],
                          field.alt,
                          field.ha,
                          field.t_to_invis,
                          field.brightness,
                          field.covered,
                          0., 0., 0., 0., 0., 0., 0.), dtype = output_dtype)
    else:
        entry = np.array((field.id,
                          float(t),
                          filter,
                          field.n_ton_visits[0]['all'],
                          field.N_visit[0]['all'],
                          field.n_ton_visits[0]['u'],
                          field.N_visit[0]['u'],
                          field.n_ton_visits[0]['g'],
                          field.N_visit[0]['g'],
                          field.n_ton_visits[0]['r'],
                          field.N_visit[0]['r'],
                          field.n_ton_visits[0]['i'],
                          field.N_visit[0]['i'],
                          field.n_ton_visits[0]['z'],
                          field.N_visit[0]['z'],
                          field.n_ton_visits[0]['y'],
                          field.N_visit[0]['y'],
                          field.cost[0][filter],
                          field.slew_t_to,
                          field.since_t_last_visit[0]['all'],
                          field.since_t_visit[0]['all'],
                          field.since_t_last_visit[0]['u'],
                          field.since_t_visit[0]['u'],
                          field.since_t_last_visit[0]['g'],
                          field.since_t_visit[0]['g'],
                          field.since_t_last_visit[0]['r'],
                          field.since_t_visit[0]['r'],
                          field.since_t_last_visit[0]['i'],
                          field.since_t_visit[0]['i'],
                          field.since_t_last_visit[0]['z'],
                          field.since_t_visit[0]['z'],
                          field.since_t_last_visit[0]['y'],
                          field.since_t_visit[0]['y'],
                          field.alt,
                          field.ha,
                          field.t_to_invis,
                          field.brightness,
                          field.covered,
                          field.F[0][filter], field.F[1][filter], field.F[2][filter], field.F[3][filter], field.F[4][filter],
                          field.F[5][filter], field.F[6][filter]), dtype = output_dtype)

    return entry

def format_output():
    output_dtype = [('Field_id', np.int),
                         ('ephemDate', np.float),
                         ('Filter', np.str_, 1),
                         ('n_ton', np.int),
                         ('n_last', np.int),
                         ('n_ton_u', np.int),
                         ('n_last_u', np.int),
                         ('n_ton_r', np.int),
                         ('n_last_r', np.int),
                         ('n_ton_i', np.int),
                         ('n_last_i', np.int),
                         ('n_ton_g', np.int),
                         ('n_last_g', np.int),
                         ('n_ton_z', np.int),
                         ('n_last_z', np.int),
                         ('n_ton_y', np.int),
                         ('n_last_y', np.int),
                         ('Cost', np.float),
                         ('Slew_t', np.float),
                         ('t_since_v_ton', np.float),
                         ('t_since_v_last', np.float),
                         ('t_since_v_ton_u', np.float),
                         ('t_since_v_last_u', np.float),
                         ('t_since_v_ton_r', np.float),
                         ('t_since_v_last_r', np.float),
                         ('t_since_v_ton_i', np.float),
                         ('t_since_v_last_i', np.float),
                         ('t_since_v_ton_g', np.float),
                         ('t_since_v_last_g', np.float),
                         ('t_since_v_ton_z', np.float),
                         ('t_since_v_last_z', np.float),
                         ('t_since_v_ton_y', np.float),
                         ('t_since_v_last_y', np.float),
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
    return output_dtype