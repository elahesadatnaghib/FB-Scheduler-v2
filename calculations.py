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
    elif field.n_ton_visits >= field.max_n_night:
        return False
    elif field.n_ton_visits == 1 and field.since_t_last_visit < field.visit_w[0]:
        return False
    elif field.n_ton_visits == 1 and field.since_t_last_visit > field.visit_w[1]:
        return False
    elif field.covered:
        return False
    if field.slew_t_to > 10 *ephem.second and field.since_t_last_visit != field.inf:
        return False
    return True

def eval_basis_fcn(field):
    F    = np.zeros(7)  # 7 is the number of basis functions
    F[0] = calculate_F1(field.slew_t_to)
    F[1] = calculate_F2(field.since_t_last_visit, field.n_ton_visits, field.t_to_invis, field.inf)
    F[2] = calculate_F3(field.since_t_visit, field.inf)
    F[3] = calculate_F4(field.alt)
    F[4] = calculate_F5(field.ha)
    F[5] = calculate_F6(field.N_visit)
    F[6] = calculate_F7(field.brightness)
    return F

def eval_cost(F, f_weight):
    return np.dot(F,f_weight)

def decision_maker(all_costs):
    winner_index = np.argmin(all_costs)
    min_cost     = all_costs[winner_index]
    #TODO check for close competitors
    return winner_index, min_cost

def eval_dt(slew_t_to, t_expo):
    return slew_t_to + t_expo

def eval_t_visit(t_decision, slew_t_to):
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
                          field.n_ton_visits,
                          field.N_visit,
                          0.,
                          0.,
                          field.since_t_last_visit,
                          field.since_t_visit,
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
                          field.n_ton_visits,
                          field.N_visit,
                          field.cost,
                          field.slew_t_to,
                          field.since_t_last_visit,
                          field.since_t_visit,
                          field.alt,
                          field.ha,
                          field.t_to_invis,
                          field.brightness,
                          field.covered,
                          field.F[0], field.F[1], field.F[2], field.F[3], field.F[4], field.F[5], field.F[6]), dtype = output_dtype)

    return entry