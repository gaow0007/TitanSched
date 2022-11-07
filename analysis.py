from client.job import (JobFactory, BaseJob, ResourceElasticJob, BatchElasticJob, HeterogeneousJob, FoundationModelJob, PreemptJob)
import options
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

opt = options.Singleton.init()

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

def parse_job_file(filename, job_type, opt): 
    df = pd.DataFrame(pd.read_csv(filename, dtype={'target_lr':str})).dropna() 
    job_list = list() 
    for idx, row in df.iterrows(): 
        row.name = row[0] # TODO
        if job_type == 'base': 
            job_instance = JobFactory(name=job_type)(name=row.name, submission_time=row.submission_time,\
                                                target_duration=row.duration, target_num_replicas=row.num_gpus, target_gpus_per_replica=1)
        elif job_type == 'foundation_model': 
            job_instance = JobFactory(name=job_type)(row, add_ckpt=opt.add_ckpt, physical=opt.physical, failure_ratio=opt.failure_ratio*0.01) 
        elif job_type == 'hpo_foundation_model': 
            job_instance = JobFactory(name=job_type)(row, add_ckpt=opt.add_ckpt, physical=opt.physical, failure_ratio=opt.failure_ratio*0.01) 
        else: 
            job_instance = JobFactory(name=job_type)(row, add_ckpt=opt.add_ckpt, physical=opt.physical, failure_ratio=opt.failure_ratio*0.01) 
        job_list.append(job_instance)
    return job_list
    
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




model_list = ['vit', 'roberta-base'] # ['roberta-base'] # , 'roberta-large'] # , 'vit', 'vit-large']
application_list = list() 
for model_idx, model in enumerate(model_list): 
    filename = 'trace/main/FM-1-{}/workload-0.csv'.format(model)
    job_type = 'foundation_model'
    job_list = parse_job_file(filename, job_type, opt)
    duration_list = list() 
    gpu_list = list() 
    service_list = list() 
    info_list = list() 
    for job in job_list: 
        duration_list.append(job.total_time_running_exclusively)
        gpu_list.append(job.target_num_gpus)
        service_list.append(duration_list[-1] * gpu_list[-1])
        info_list.append((service_list[-1], job.application.scale, job.application.name))
        if job.application.name not in application_list: 
            application_list.append(job.application.name)
            print('name == {}'.format(job.application.name))
            for gpu in [1, 2, 4, 8, 16]: 
                if gpu <= job.max_num_gpus: 
                    print("gpu {}, speedup {}".format(gpu, job.predict_remaining_time(1) / job.predict_remaining_time(gpu)))

    info_list = sorted(info_list)
    # for info in info_list: print(info)
    # continue 
    # import pdb; pdb.set_trace() 
        
    x_list, y_list = get_cdf(service_list)
    plt.plot(x_list, y_list, color=color_list[model_idx], label=model)
plt.legend() 
plt.savefig('service.jpg'.format())