__author__ = 'Elahe'

import numpy as np
import ephem
from operator import attrgetter
from MultiProposalCalculations import *

sciences = ["WFD", "DD", "GP", "NE", "SE"]

def eval_init_state(fields, suggestion, manual = False):        # TODO Feasibility of the initial field needs to be checked
    if manual:
        return suggestion
    else:
        #build a vector of all altitudes at t start
        all_alt_start = [field.all_moments_data[0]['alts'] for field in fields]
        winner_index = np.argmax(all_alt_start)
        return fields[winner_index]

def eval_init_filter(filters):
    return filters[0]

def eval_feasibility(field):
    if not basic_feasible(field):
        return False
    science = detect_science(field)
    if science == "WFD":
        if not universalsurvey_feasible(field):
            return False
    elif science == "DD":
        if not DDsurvey_feasible(field):
            return False
    return True

def eval_feasibility_filter(filter, current_filter, current_field):
    science = detect_science(current_field)
    if science == "WFD":
        if not universalsurvey_filter_feasible(filter, current_filter):
            return False
    elif science == "DD":
        if not DDsurvey_filter_feasible(filter, current_field):
            return False
    return True

def eval_basis_fcn(field, curr_filter_name, filters):
    F    = np.zeros(7, dtype = [('u', np.float),('g', np.float),('r', np.float),('i', np.float),('z', np.float),('y', np.float)])  # 7 is the number of basis functions
    science = detect_science(field)
    for f in filters:
        if f.feasible:
            F[0][f.name] = calculate_F1(field.slew_t_to, f.name, curr_filter_name)
            F[1][f.name] = calculate_F2(field.since_t_last_visit[0][f.name], field.n_ton_visits[0][f.name], field.t_to_invis, field.inf, science, field.n_ton_visits[0]['all'])
            F[2][f.name] = calculate_F3(filters, f.name)
            F[3][f.name] = calculate_F4(field.alt, f.name)
            F[4][f.name] = calculate_F5(field.ha, f.name)
            F[5][f.name] = calculate_F6(field.N_visit['all'][0], field.Max_N_visit['all'], field.N_visit[0][f.name], field.Max_N_visit[0][f.name])
            F[6][f.name] = calculate_F7(field.brightness,field.moonsep, f.name)
        else:
            F[0][f.name] = None
            F[1][f.name] = None
            F[2][f.name] = None
            F[3][f.name] = None
            F[4][f.name] = None
            F[5][f.name] = None
            F[6][f.name] = None
    return F

def eval_cost(F, f_weight, filters):
    c = np.zeros(1, dtype = [('u', np.float),('g', np.float),('r', np.float),('i', np.float),('z', np.float),('y', np.float)])
    for f in filters:
        if f.feasible:
            c[0][f.name] = np.dot(F[:][f.name],f_weight)
        else:
            c[0][f.name] = f.inf
    return c

def decision_maker(all_costs):
    min_cost  = np.zeros(6, dtype= np.int)
    min_index = np.zeros(6, dtype= np.int)

    winner_filter_index = 0
    for index,f in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
        min_index[index] = np.argmin(all_costs[:][f])
        min_cost[index]  = np.min(all_costs[:][f])

        if min_cost[index] < min_cost[winner_filter_index]:
            winner_filter_index = index
    winner_index = min_index[winner_filter_index]
    winner_cost  =  min_cost[winner_filter_index]
    #TODO check for close competitors
    return winner_index, winner_cost, winner_filter_index

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


def calculate_F1(slew_t_to, filter, curr_filter):            # slew time cost 0~2
    normalized_slew = (slew_t_to /ephem.second) /5
    if filter == curr_filter:
        return normalized_slew
    else:
        return normalized_slew + 4 # count for filter change time cost

def calculate_F2(since_t_last_visit, n_ton_visits, t_to_invis, inf, science, n_ton_visits_all):   # night urgency 0~10
    if science == "DD" and n_ton_visits_all > 0:
        return -inf
    if since_t_last_visit == inf:
        filter_indep =  5
    if n_ton_visits == 2:
        filter_indep = 10
    elif n_ton_visits == 1:
        if t_to_invis < 30 * ephem.minute:
            filter_indep =  0
        else:
            filter_indep = 5 * (1 - np.exp(-1* since_t_last_visit / 20 * ephem.minute))
    return filter_indep

def calculate_F3(filters, f_name):       # filter urgency 0~1
    index = ['u','g','r','i','z','y'].index(f_name)
    max_n_visit_in = max(f.n_visit_in for f in filters)
    urgency = float(filters[index].n_visit_in) / (max_n_visit_in +1)
    return filters[index].n_changed_to + urgency


def calculate_F4(alt, filter_name):              # altitude cost 0~1
    #filter_indep_alt_cost = (1./(1-np.cos(alt))) -1 # 0~2.5
    #filter_indep_alt_cost = 1- alt/np.pi
    #filter_corr_alt_cost  = corr_alt_for_filter(filter_indep_alt_cost, filter_name)
    filter_corr_alt_cost   = alt_allocation(alt, filter_name)
    return filter_corr_alt_cost

def calculate_F5(ha, filter):               # hour angle cost 0~1
    return np.abs(ha)

def calculate_F6(N_visit_tot, Max_N_visit, N_visit_filter, Max_N_visit_filter):   # overall urgency 0~2
    return float(N_visit_tot)/(Max_N_visit +1) + float(N_visit_filter)/(Max_N_visit_filter +1)  # normalized n_visit +1 to make sure won't have division by 0

def calculate_F7(brightness, moonsep, filter_name):       # normalized brightness 0~1 #TODO has to go to the constraints
    if moonsep < np.deg2rad(25) and filter_name == 'u':
        return 1e10
    if moonsep < np.deg2rad(20) and filter_name == 'g':
        return 1e10
    if moonsep < np.deg2rad(15) and filter_name == 'r':
        return 1e10
    if moonsep < np.deg2rad(10) and filter_name == 'i':
        return 1e10
    if moonsep < np.deg2rad(5) and filter_name == 'z':
        return 1e10
    if moonsep < np.deg2rad(0) and filter_name == 'y':
        return 1e10
    return brightness



# miscellaneous
def record_assistant(field, t, filter_name, output_dtype, first_entry = False):
    if first_entry:
        entry = np.array((field.id,
                          float(t),
                          filter_name,
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
                          filter_name,
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
                          field.cost[0][filter_name],
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
                          field.F[0][filter_name], field.F[1][filter_name], field.F[2][filter_name], field.F[3][filter_name],
                          field.F[4][filter_name], field.F[5][filter_name], field.F[6][filter_name]), dtype = output_dtype)

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


def corr_alt_for_filter(x, filter_name):
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    break_point = 0.2
    # values at 0, breakpoint, and 1
    vals = np.array([(0,0.1,4.5),(.2,.4,4.4),(.4,.7,4.3),(.6,1.0,4.2),(.8,1.3,4.1),(1,1.6,4)])
    for index,f in enumerate(filters):
        if filter_name == f:  # 2piece linear
            if x <= break_point:
                a = (vals[index][1] - vals[index][0])/ break_point
                b = vals[index][0]
                y = a*x + b
            else:
                a = (vals[index][2] - vals[index][1])/ (1-break_point)
                b = vals[index][1]
                y = a*x + b
            return y

def alt_allocation(alt, filter_name):
    n_alt = 2*alt/np.pi
    index = ['u', 'g', 'r', 'i', 'z', 'y'].index(filter_name)
    traps = np.array([0.95,0.85,0.75,0.65,0.55,0.45])
    if n_alt > 0.95:
        n_alt = 0.95
    if n_alt < 0.45:
        n_alt = 0.45
    return 100*np.square(n_alt-traps[index]) + ((1./(1-np.cos(alt))) -1) * 5


