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

''' Universal '''
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

''' DD cosmology '''
def is_DD(field_id): #TODO temporarily just by id, later by label or location
    if field_id in [744, 2412, 1427, 2786, 290]:
        return True
    return False

def DDsurvey_feasible(field):
    if field.n_ton_visits[0]['all'] >= 6: # with 6 visits in all filters, observation is done for the field
        return False
    if field.N_visit> 0 and field.since_t_visit <= 2: # if field is observed witgin two previous nights, it's infeasible
        return False
    if field.t_to_invis < (6 - field.n_ton_visits['all']) * field.t_expo: # if there is not enough time to finish the DD observation
        return False
    return True

def DDsurvey_filter_feasible(filter, current_field):
    if current_field.n_ton_visits[0]['all'] < 6 and current_field.n_ton_visits[filter.name] != 0: # so next observation would be on the same DD field
        return False
    return True

''' Galactic plane '''

''' North ecliptic '''

'''South celestial pole '''




