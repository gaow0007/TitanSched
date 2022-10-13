import numpy as np 

x_list = [1280000, 50000, 1030]
log_x_list = [np.log(x) for x in x_list]
# y_list = [10, 100, 235]
y_list = [20000, 10000, 500]
a1, a2 = np.polyfit(log_x_list, y_list, deg=1)
for x in [1000, 5000, 10000, 50000, 100000]: 
# for x in x_list: 
    log_x = np.log(x)
    print(x, a2 + a1 * log_x)
import pdb; pdb.set_trace() 