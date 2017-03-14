import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import ephem


# 1 : N_visit
# 2 : slew_t_to
# 3 : alt
# 5 : moon sep
# 6 : n_ton_visits
# 7 : since_t_last_visit
# 8~13: cost
target = 1
step_index = 4
steps_tp_save        = [0,10,11,12,500]
all_filter = False

step = steps_tp_save[step_index]
all_data   = np.load('Watch/Data{}.npy'.format(step))
data_indices = np.nonzero(all_data[:,0])
last_row_index = data_indices[-1][-1]

summary    = all_data[last_row_index,:]
best_filter= summary[3]
best_field = summary[2] +1
best_index = np.where(all_data[0:last_row_index -1,0] == best_field)[0][0]

if all_filter:
    D = np.zeros((1,6*(last_row_index -1)))
    for i in range(6):
        D[0, i*(last_row_index -1):(i+1)*(last_row_index -1)] = all_data[0:last_row_index -1,target+i]
        # the histogram of the data
    D = np.transpose(D)
    max_factor = np.max(D)
    D = np.divide(D, max_factor)
    n, bins, patches = plt.hist(D, 50, facecolor='green', alpha=0.75)
    print(np.divide(all_data[best_index,target+best_filter],max_factor))

    plt.ylabel('Frequency')
    if target == 8:
        plt.xlabel('Cost(field,filter)')
    if target == 14:
        plt.xlabel('F1(field,filter)')
    if target == 20:
        plt.xlabel('F2(field,filter)')
    if target == 26:
        plt.xlabel('F3(field,filter)')
    if target == 32:
        plt.xlabel('F4(field,filter)')
    if target == 38:
        plt.xlabel('F5(field,filter)')

else:

    D1 = all_data[0:last_row_index -1,target]
    max_factor = np.max(D1)
    D1 = np.divide(D1,max_factor)

    print(np.divide(all_data[best_index,target],max_factor))
    filters = ['u','g','r','i','z','y']
    print('best field: {}, best filter: {}'.format(best_filter, filters[np.int(best_filter)]))


    plt.bar(all_data[0:last_row_index -1, 0],D1, color = 'b')
    plt.axvline(x = best_field, color='orange', linestyle='--', linewidth = 3)


    #plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    plt.axis([1, 4026, 0, 1])
    plt.grid(True)
    plt.xlabel('Fields ID')

    if target == 1:
        plt.ylabel('Normalized number of visits')
    if target == 2:
        plt.ylabel('Normalized slew time to')
    if target == 3:
        plt.ylabel('Normalized altitude')
    if target == 5:
        plt.ylabel('Normalized separation with the moon')
    if target == 6:
        plt.ylabel('Normalized time passed since last visit')




plt.show()