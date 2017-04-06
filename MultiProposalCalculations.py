__author__ = 'Elahe'

import numpy as np
import ephem

sciences = ["WFD", "DD", "GP", "NE", "SE"]

''' basic basis functions '''
def detect_region(field):
    if field.label == 'DD':
        return 'DD'
    if field.label == 'GP':
        return 'GP'
    if field.label == 'NES':
        return 'NES'
    if field.label == 'SCP':
        return 'SCP'
    return "WFD"

def basic_F1(slew_t_to, filter_name, curr_filter_name):           # slew time cost 0~2
    normalized_slew = (slew_t_to /ephem.second) /5
    if filter_name == curr_filter_name:
        return normalized_slew
    return normalized_slew + 5 # count for filter change time cost

def basic_F2(n_ton_visits):
    if n_ton_visits == 0:
        return 5
    if n_ton_visits == 1:
        return 5
    if n_ton_visits == 2:
        return 10

def basic_F3(filters, f_name):
    index = ['u','g','r','i','z','y'].index(f_name)
    max_n_visit_in = max(f.n_visit_in for f in filters) #for the night
    night_urgency  = float(filters[index].n_visit_in) / (max_n_visit_in +1)
    max_N_visit_in = max(f.N_visit_in for f in filters) # overall
    overall_urgency= 5. / (filters[index].N_visit_in - max_N_visit_in +1)
    return filters[index].n_changed_to + night_urgency + overall_urgency

def basic_F4(alt, filter_name, dec):
    max_alt = -0.517781017 + np.deg2rad(90) - dec  # -0.517781017 is the LSST site's latitude
    if dec < -0.517781017:
        max_alt = 0.517781017 + np.deg2rad(90) + dec
    normalized_alt = float(alt)/max_alt
    filter_corr_alt_cost   = alt_allocation(normalized_alt, filter_name)
    return filter_corr_alt_cost

def basic_F5(ha):               # hour angle cost 0~1
    return np.abs(ha)

def basic_F6(N_visit_tot, Max_N_visit, N_visit_filter, Max_N_visit_filter):
    return 5./(Max_N_visit - N_visit_tot +1) + 5. / (Max_N_visit_filter - N_visit_filter + 1)


def basic_F7(brightness, moonsep, filter_name):       # has to change for brightness only and the output is actual cost (not infinity)
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

''' Basis functions '''
def calculate_F1(region, **kwargs): #  time cost
    slew_t_to        = kwargs.get('slew_t_to')
    filter_name      = kwargs.get('filter_name')
    curr_filter_name = kwargs.get('curr_filter_name')
    if region == 'WFD':
        return basic_F1(slew_t_to, filter_name, curr_filter_name)
    if region == 'DD':
        return basic_F1(slew_t_to, filter_name, curr_filter_name)
    if region == 'GP':
        return basic_F1(slew_t_to, filter_name, curr_filter_name)
    if region == 'NES':
        return basic_F1(slew_t_to, filter_name, curr_filter_name)
    if region == 'SCP':
        return basic_F1(slew_t_to, filter_name, curr_filter_name)

def calculate_F2(region, **kwargs): # night urgency of a field
    n_ton_visits = kwargs.get('n_ton_visits')
    if region == 'WFD':
        t_to_invis         = kwargs.get('t_to_invis')
        since_t_last_visit = kwargs.get('since_t_last_visit')
        if n_ton_visits == 1:   # to decrease the cost for the same-night second visit that is required for WFD
            if t_to_invis < 30 * ephem.minute or since_t_last_visit > 30 * ephem.minute:
                night_urgency =  0
            else:
                night_urgency = 5 * (1 - np.exp(-1* since_t_last_visit / 20 * ephem.minute))
            return night_urgency
        return basic_F2(n_ton_visits)
    if region == 'DD':
        inf         = kwargs.get('inf')
        N_visit_tot = kwargs.get('N_visit_tot'); N_visit_filter = kwargs.get('N_visit_filter')
        if n_ton_visits == 0:
            if N_visit_tot % 6 == 0:    #when DD observation hasn't started yet, we treat DD field like a basic field
                return basic_F2(n_ton_visits)
            if N_visit_tot % 6 != 0:    #we need to complete the previous night's DD observation
                if N_visit_filter >= float(N_visit_tot)/3: # the first filter for the second DD night would be a new one
                    return inf # so the next visit would be with a different filter
                return 0
        return -inf
    if region == 'GP':
        return basic_F2(n_ton_visits)
    if region == 'NES':
        t_to_invis         = kwargs.get('t_to_invis')
        since_t_last_visit = kwargs.get('since_t_last_visit')
        if n_ton_visits == 1:   # to decrease the cost for the same-night second visit that is required for NES
            if t_to_invis < 30 * ephem.minute or since_t_last_visit > 30 * ephem.minute:
                night_urgency =  0
            else:
                night_urgency = 5 * (1 - np.exp(-1* since_t_last_visit / 20 * ephem.minute))
            return night_urgency
        return basic_F2(n_ton_visits)
    if region == 'SCP':
        return basic_F2(n_ton_visits)

def calculate_F3(region, **kwargs):  # night and overall filter urgency
    filters = kwargs.get('filters')
    f_name  = kwargs.get('f_name')
    if region == 'WFD':
        return basic_F3(filters, f_name)
    if region == 'DD':
        n_ton_visits = kwargs.get('n_ton_visits')
        if n_ton_visits == 0:
            return basic_F3(filters, f_name)
        return 0
    if region == 'GP':
        return basic_F3(filters, f_name)
    if region == 'NES':
        index = ['u','g','r','i','z','y'].index(f_name)
        intrinsic_urgency = [100, 0, 0, 0, 0, 100]   # not in the constraints because we might want u,y observations later
        return basic_F3(filters, f_name) + intrinsic_urgency[index]
    if region == 'SCP':
        return basic_F3(filters, f_name)

def calculate_F4(region, **kwargs):     # filter dependent altitude cost
    alt         = kwargs.get('alt')
    filter_name = kwargs.get('filter_name')
    dec         = kwargs.get('dec')
    if region == 'WFD':
        return basic_F4(alt, filter_name, dec) + ((1./(1-np.cos(alt))) -1) * 4
    if region == 'DD':
        n_ton_visits = kwargs.get('n_ton_visits') + ((1./(1-np.cos(alt))) -1) * 6 # to catch the DD fields in a better location
        if n_ton_visits == 0:
            return basic_F4(alt, filter_name, dec) + ((1./(1-np.cos(alt))) -1) * 4
        else:
            return 0
    if region == 'GP':
        return basic_F4(alt, filter_name, dec) + ((1./(1-np.cos(alt))) -1) * 4
    if region == 'NES':
        return basic_F4(alt, filter_name, dec) + ((1./(1-np.cos(alt))) -1) * 4
    if region == 'SCP':
        return basic_F4(alt, filter_name, dec) + ((1./(1-np.cos(alt))) -1) * 4

def calculate_F5(region, **kwargs): # hour angle cost
    ha = kwargs.get('ha')
    if region == 'WFD':
        return basic_F5(ha)
    if region == 'DD':
        return basic_F5(ha)
    if region == 'GP':
        return basic_F5(ha)
    if region == 'NES':
        return basic_F5(ha)
    if region == 'SCP':
        return basic_F5(ha)

def calculate_F6(region, **kwargs):  # overall urgency of a field-filter
    N_visit_tot    = kwargs.get('N_visit_tot');    Max_N_visit = kwargs.get('Max_N_visit')
    N_visit_filter = kwargs.get('N_visit_filter'); Max_N_visit_filter = kwargs.get('Max_N_visit_filter')
    if region == 'WFD':
        return basic_F6(N_visit_tot, Max_N_visit, N_visit_filter, Max_N_visit_filter)
    if region == 'DD':
        return basic_F6(N_visit_tot, Max_N_visit, N_visit_filter, Max_N_visit_filter)
    if region == 'GP':
        return basic_F6(N_visit_tot, Max_N_visit, N_visit_filter, Max_N_visit_filter)
    if region == 'NES':
        return basic_F6(N_visit_tot, Max_N_visit, N_visit_filter, Max_N_visit_filter)
    if region == 'SCP':
        return basic_F6(N_visit_tot, Max_N_visit, N_visit_filter, Max_N_visit_filter)

def calculate_F7(region, **kwargs):
    brightness = kwargs.get('brightness')
    moonsep    = kwargs.get('moonsep')
    filter_name= kwargs.get('filter_name')
    if region == 'WFD':
        return basic_F7(brightness, moonsep, filter_name)
    if region == 'DD':
        return basic_F7(brightness, moonsep, filter_name)
    if region == 'GP':
        return basic_F7(brightness, moonsep, filter_name)
    if region == 'NES':
        return basic_F7(brightness, moonsep, filter_name)
    if region == 'SCP':
        return basic_F7(brightness, moonsep, filter_name)


''' feasibility '''
def eval_feasibility(field):
    if not basic_feasible(field):
        return False
    if field.label == "WFD":
        if not WFD_feasible(field):
            return False
    if field.label == "DD":
        if not DD_feasible(field):
            return False
    if field.label == "GP":
        if not GP_feasible(field):
            return False
    if field.label == "NES":
        if not NES_feasible(field):
            return False
    if field.label == "SCP":
        if not SCP_feasible(field):
            return False
    return True

def basic_feasible(field):
    if not field.visible:
        return False
    elif field.covered:
        return False
    if field.slew_t_to > 5 *ephem.second and field.since_t_last_visit[0]['all'] != field.inf:
        return False
    return True

def WFD_feasible(field):
    if field.n_ton_visits[0]['all'] >= field.max_n_night:
        return False
    #elif field.n_ton_visits[0]['all'] == 1 and field.since_t_last_visit[0]['all'] < field.visit_w[0]:
    elif field.n_ton_visits[0]['all'] != 0 and field.since_t_last_visit[0]['all'] < field.visit_w[0]:
        return False
    #elif field.n_ton_visits[0]['all'] == 1 and field.since_t_last_visit[0]['all'] > field.visit_w[1]:
    elif field.n_ton_visits[0]['all'] != 0 and field.since_t_last_visit[0]['all'] > field.visit_w[1]:
        return False
    return True

def DD_feasible(field):
    if field.n_ton_visits[0]['all'] >= 3: # with 3 visits in all filters, observation is done for the field
        return False
    if field.N_visit['all']%6 == 0 and field.N_visit> 0 and field.since_t_visit <= 2: # if field is observed in 6 filters within two previous nights, it's infeasible
        return False
    if field.n_ton_visits[0]['all'] == 0 and field.t_to_invis < 6* field.t_expo: # if there is not enough time to finish the DD observation
        return False
    return True

def GP_feasible(field):
    if field.n_ton_visits[0]['all'] >= 1:#field.max_n_night:
        return False
    return True

def NES_feasible(field):
    if field.n_ton_visits[0]['all'] >= 1:#field.max_n_night:
        return False
    #elif field.n_ton_visits[0]['all'] == 1 and field.since_t_last_visit[0]['all'] < field.visit_w[0]:
    elif field.n_ton_visits[0]['all'] != 0 and field.since_t_last_visit[0]['all'] < field.visit_w[0]:
        return False
    #elif field.n_ton_visits[0]['all'] == 1 and field.since_t_last_visit[0]['all'] > field.visit_w[1]:
    elif field.n_ton_visits[0]['all'] != 0 and field.since_t_last_visit[0]['all'] > field.visit_w[1]:
        return False
    return True

def SCP_feasible(field):
    if field.n_ton_visits[0]['all'] >= field.max_n_night:
        return False

#filter feasibility
def eval_feasibility_filter(filter, current_filter, current_field):
    if not basic_filter_feasible(filter, current_filter) and current_field.label != "DD":
        return False
    elif current_field.label == "DD":
        if not DDsurvey_filter_feasible(filter, current_field):
            return False
    return True

def basic_filter_feasible(filter, current_filter):
    if (filter.name != current_filter.name) and current_filter.n_current_batch <= 30:
        return False
    return True

def DDsurvey_filter_feasible(filter, current_field):
    if current_field.n_ton_visits[0]['all'] < 3: # so deep drilling of the current night of the field is not over yet
        if current_field.n_ton_visits[0][filter.name] != 0: # so next observation would be with a different filter
            return False
        if current_field.N_visit['all'][0] % 6 != 0: # so this is the second night that we expect three different filter compared to the last night
            temp = np.array([current_field.N_visit['u'][0], current_field.N_visit['g'][0], current_field.N_visit['r'][0], current_field.N_visit['i'][0],
                             current_field.N_visit['z'][0], current_field.N_visit['y'][0]])
            sort_visit_filter = np.sort(temp)
            print(temp, sort_visit_filter,current_field.N_visit[filter.name], sort_visit_filter[3])
            if current_field.N_visit[filter.name][0] >= sort_visit_filter[3]:
                return False
    return True




# miscellaneous

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

def alt_allocation(normalized_alt, filter_name):
    index = ['u', 'g', 'r', 'i', 'z', 'y'].index(filter_name)
    traps = np.array([0.95,0.85,0.75,0.65,0.55,0.45])
    if normalized_alt > 0.95:
        normalized_alt = 0.95
    if normalized_alt < 0.45:
        normalized_alt = 0.45
    return 80*np.square(normalized_alt-traps[index])



