import os, sys
import csv
import yaml 
from easydict import EasyDict
import random 
import numpy as np 
from datetime import datetime
import pandas as pd
import ast 

import os, sys
import matplotlib.pyplot as plt 
import seaborn
import matplotlib
import numpy as np 
import math 

color_list = ['tab:orange',
            'tab:blue',
            'tab:green',
            'tab:red',
            'tab:purple',
            'tab:brown',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'tab:cyan']


def get_cdf(data):
    """Returns the CDF of the given data.
    
       Args:
           data: A list of numerical values.
           
       Returns:
           An pair of lists (x, y) for plotting the CDF.
    """
    sorted_data = sorted(data)
    p = 100. * np.arange(len(sorted_data)) / (len(sorted_data) - 1)
    return sorted_data, p


if __name__ == '__main__': 
    trace_root = 'all_trace'
    for trace_idx, trace in enumerate(['BM', 'Helios', 'MLaas', 'Philly']): 
        filename = os.path.join(trace_root, trace + '.csv')
        columns = ['submission_time', 'duration', 'num_gpus']
        df = pd.DataFrame(pd.read_csv(filename), columns=columns).dropna() 
        duration_list = df.duration.tolist() 
        gpu_list = df.num_gpus.tolist() 
        service_list = [duration for duration, gpu in zip(duration_list, gpu_list)]
        x_list, y_list = get_cdf(service_list)
        plt.plot(x_list, y_list, color=color_list[trace_idx], label=trace)
    plt.legend() 
    plt.savefig('{}.jpg'.format(trace_root))


