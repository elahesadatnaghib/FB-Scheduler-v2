__author__ = 'Elahe'

import numpy as np
import ephem

sciences = ["WFD", "DD", "GP", "NE", "SE"]

''' basic functions '''
def basic_feasible(field):
    if not field.visible:
        return False
    elif field.covered:
        return False
    if field.slew_t_to > 5 *ephem.second and field.since_t_last_visit[0]['all'] != field.inf:
        return False
    return True

def detect_science(field):
    if is_DD(field.id):
        return "DD"
    return "WFD"

''' Universal WFD '''
def universalsurvey_feasible(field):
    if field.n_ton_visits[0]['all'] >= field.max_n_night:
        return False
    #elif field.n_ton_visits[0]['all'] == 1 and field.since_t_last_visit[0]['all'] < field.visit_w[0]:
    elif field.n_ton_visits[0]['all'] != 0 and field.since_t_last_visit[0]['all'] < field.visit_w[0]:
        return False
    #elif field.n_ton_visits[0]['all'] == 1 and field.since_t_last_visit[0]['all'] > field.visit_w[1]:
    elif field.n_ton_visits[0]['all'] != 0 and field.since_t_last_visit[0]['all'] > field.visit_w[1]:
        return False
    return True

def universalsurvey_filter_feasible(filter, current_filter):
    if (filter.name != current_filter.name) and current_filter.n_current_batch <= 30:
        return False
    return True

def calculate_F2_WFD(since_t_last_visit, n_ton_visits, t_to_invis, inf):
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

def calculate_F6_WFD(N_visit_tot, Max_N_visit, N_visit_filter, Max_N_visit_filter):
    return float(N_visit_tot)/(Max_N_visit +1) + float(N_visit_filter)/(Max_N_visit_filter +1)+ 5  # normalized n_visit +1 to make sure won't have division by 0


''' DD cosmology '''
def is_DD(field_id): #TODO temporarily just by id, later by label or location
    if field_id in [744, 2412, 1427, 2786, 290]:
        return True
    return False

def DDsurvey_feasible(field):
    if field.n_ton_visits[0]['all'] >= 3: # with 3 visits in all filters, observation is done for the field
        return False
    if field.N_visit['all']%6 == 0 and field.N_visit> 0 and field.since_t_visit <= 2: # if field is observed in 6 filters within two previous nights, it's infeasible
        return False
    if field.n_ton_visits[0]['all'] == 0 and field.t_to_invis < (3 - field.n_ton_visits['all']) * field.t_expo: # if there is not enough time to finish the DD observation
        return False
    return True

def DDsurvey_filter_feasible(filter, current_field):
    if current_field.n_ton_visits[0]['all'] < 3: # so deep drilling of the current night of the field is not over yet
        if current_field.n_ton_visits[0][filter.name] != 0: # so next observation would be with a different filter
            return False
        if current_field.N_visit['all'] % 6 != 0: # so this is the second night that we expect three different filter compared to the last night
            temp = np.array([current_field.N_visit['u'], current_field.N_visit['g'], current_field.N_visit['r'], current_field.N_visit['i'],
                             current_field.N_visit['z'], current_field.N_visit['y']])
            sort_visit_filter = np.sort(temp)
            print (temp)
            if current_field.N_visit[filter.name] >= sort_visit_filter[3]:
                return False
    print(current_field.id, filter.name, current_field.n_ton_visits[0])
    return True

def calculate_F2_DD(since_t_last_visit, n_ton_visits, t_to_invis, n_ton_visits_all, inf):
    if n_ton_visits_all > 0: # so the next observation would be on the same DD field
        return -inf
    return calculate_F2_WFD(since_t_last_visit, n_ton_visits, t_to_invis, inf)

def calculate_F6_DD(N_visit_tot, Max_N_visit, N_visit_filter, Max_N_visit_filter, inf):
    if N_visit_tot % 6 != 0: #we need to complete the previous night's DD observation
        if N_visit_filter >= N_visit_tot/3: # the first filter for the second DD night would be a new one
            return inf
        return -3
    return calculate_F6_WFD(N_visit_tot, Max_N_visit, N_visit_filter, Max_N_visit_filter)

''' Galactic plane '''

''' North ecliptic spur'''

'''South celestial pole '''




