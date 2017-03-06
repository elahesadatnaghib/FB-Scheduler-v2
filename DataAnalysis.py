import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import ephem
import FBDE
import time


steps_tp_save        = [0,10,11,12]

step = steps_tp_save[1]
all_data = np.load('Watch/Data{}.npy'.format(step))
night    = all_data[-1,:]
best_field = night[2]
F1 = all_data[0:785,2] /ephem.second
print(np.max(F1), np.min(F1))


print(all_data[best_field -1])



# the histogram of the data
n, bins, patches = plt.hist(F1, 50,facecolor='green', alpha=0.75)



plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()