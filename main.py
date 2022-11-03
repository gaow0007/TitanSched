
import os, sys
import csv, json
from alg.scheduler.edf import EDFScheduler
from alg.scheduler.titan import TitanScheduler
from alg.scheduler.hpo_titan import HPOTitanScheduler
from client.job.base import JobManager
import numpy as np
import pandas as pd
import multiprocessing
import options
import copy, glob
from alg import (PlaceMentFactory, YarnCSScheduler, TiresiasScheduler, \
                GandivaScheduler, ThemisScheduler, ShortestRemainingTimeFirstScheduler, \
                TetriSchedScheduler, SigmaScheduler, GenieScheduler, OptimusScheduler, PolluxScheduler)

from server import cluster, meta
from client.job import (JobFactory, BaseJob, ResourceElasticJob, BatchElasticJob, HeterogeneousJob, FoundationModelJob, PreemptJob)
from client.job.state import JobState, DDLState
from client.user import (UserManager, VanillaUser, TimeAwareUser) 

from utils.logger import getLogger

import random
import numpy as np
seed=42
random.seed(seed)
np.random.seed(seed)

# USERS = users.USERS
# JOBS = jobs.JOBS
# CLUSTER = cluster.CLUSTER
# ProfileManager = profiler.ProfileManager


def prepare_job_manager(): 
    return JobManager() 

def parse_job_file(filename, job_type, job_manager, opt): 
    df = pd.DataFrame(pd.read_csv(filename, dtype={'target_lr':str})).dropna() 
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
        
        if hasattr(row, 'ddl_time_list'): 
            if isinstance(row.ddl_time_list, str): 
                job_instance.ddl_time_list = [int(item) for item in row.ddl_time_list.split('-')]
                job_instance.ddl_value_list = [int(item) for item in row.ddl_value_list.split('-')]
            elif isinstance(row.ddl_time_list, int): 
                job_instance.ddl_time_list =  [row.ddl_time_list]
                job_instance.ddl_value_list =  [row.ddl_value_list]

            job_instance.ddl_time = job_instance.ddl_time_list[0] 
            for ddl_type in DDLState: 
                if ddl_type.value == row.ddl_type: 
                    job_instance.ddl_type = ddl_type 
                    break

            if not hasattr(job_instance, 'ddl_type'): 
                raise NotImplementedError 
        # print(job_instance.ddl_type, job_instance.ddl_type==DDLState.STRICT, job_instance.ddl_type==DDLState.BEST)
        job_manager.submit_job(job_instance)
        
    job_manager.prepare_job_start_events()


def prepare_partion_size(filename):
    user_partition_size = dict()
    share = pd.read_csv(filename)
    for index, row in share.iterrows():
        user_partition_size[row['user_name']] = int(int(row['partition_size']) * 0.6)
    
    return user_partition_size
       

def prepare_cluster(opt):
    if opt.num_node_p_switch == -1:
        partition_size_info = prepare_partion_size(opt.user_partition_size)
        opt.num_node_p_switch = partition_size_info[opt.user_partition] 

    cluster_info = list() 
    if opt.heter: 
        
        num_node_p_gpu_kind = opt.num_node_p_switch // len(opt.heter_gpus)
        for heter_id, heter_gpu in enumerate(sorted(opt.heter_gpus)): 
            cluster_info.append((heter_id, heter_gpu, num_node_p_gpu_kind))
    else: 
        cluster_info.append((0, 'V100', opt.num_node_p_switch))

    cluster_instance_info = dict()
    
    for heter_id, gpu_name, num_node_p_switch in cluster_info: 
        cluster_instance = cluster.Cluster(num_switch=opt.num_switch, 
                                num_node_p_switch=num_node_p_switch, 
                                num_gpu_p_node=opt.num_gpu_p_node,
                                num_cpu_p_node=opt.num_cpu_p_node, 
                                mem_p_node=opt.mem_p_node)

        cluster_instance.init_infra(num_switch=opt.num_switch, 
                            num_node_p_switch=num_node_p_switch, 
                            num_gpu_p_node=opt.num_gpu_p_node, 
                            num_cpu_p_node=opt.num_cpu_p_node, 
                            mem_p_node=opt.mem_p_node, 
                            gpu_kind=gpu_name,
                            )
        cluster_instance_info[gpu_name] = cluster_instance
    return meta.MetaCluster(cluster_instance_info)


def prepare_cluster_shadow():
    shadow_cluster = cluster._Cluster()
    shadow_cluster.init_infra(num_switch=opt.num_switch, 
                        num_node_p_switch=opt.num_node_p_switch, 
                        num_gpu_p_node=0, 
                        num_cpu_p_node=opt.num_cpu_p_node, 
                        mem_p_node=opt.mem_p_node)
    return shadow_cluster


def prepare_user(namelist):
    if os.path.exists(namelist): 
        with open(namelist, 'r') as f:
            names = f.readlines()
            for name in names:
                if opt.schedule == 'time-aware':
                    new_user = TimeAwareUser(JOBS=JOBS, CLUSTER=CLUSTER, name=name.strip(), logger=logger, quota=50)
                else:
                    new_user = VanillaUser(JOBS=JOBS, CLUSTER=CLUSTER, name=name.strip(), logger=logger)
                USERS.add_user(new_user)
    else: 
        new_user = VanillaUser(JOBS=JOBS, CLUSTER=CLUSTER, name='single-cluster', logger=logger)
        USERS.add_user(new_user)


def summary_all_jobs(job_manager):
    assert all([job.status == JobState.END for job in job_manager.job_list])
    num_job = 1.0 * len(job_manager.job_list)
    jct = 0
    min_time = sys.maxsize
    max_time = 0
    for job in job_manager.job_list:
        jct += (job.completion_time - job.submission_time) / num_job
        min_time = min(job.submission_time, min_time)
        max_time = max(job.completion_time, max_time)
    preempt = 0
    for job in job_manager.job_list: 
        preempt += job.num_restarts

    logger.info('average jct of scheduler %s is %d, %02f (hour)'%(opt.schedule,  jct, jct / 3600))
    logger.info('makespan of scheduler %s is %d, %02f (hour)'%(opt.schedule,  max_time - min_time, (max_time - min_time) / 3600))
    logger.info('total num_restarts of scheduler %s is %d'%(opt.schedule,  preempt))
    if len(job_manager.job_list) > 0: 
        job_template = job_manager.job_list[0]
        if hasattr(job_template, 'ddl_time_list'): 
            slo_number, slo_metric = 0, 0
            best_number, best_metric = 0, 0
            for job in job_manager.job_list: 
                if job.ddl_time_list[0] == job.submission_time: 
                    best_number += 1 
                    best_metric +=  (job.completion_time - job.submission_time)
                elif len(job.ddl_time_list) == 1: 
                    slo_number += 1 
                    if job.completion_time > job.ddl_time_list[0]: 
                        slo_metric += 1
                else: 
                    slo_number += 1
                    max_reward = max(job.ddl_value_list)
                    added = False 
                    for ddl_time, ddl_reward in zip(job.ddl_time_list. job.ddl_value_list): 
                        if job.completion_time <= ddl_time: 
                            slo_metric += 1 - 1.0 * ddl_reward / max_reward
                            added = True 
                            break 
                    if not added: 
                        slo_metric += 1
            
            logger.info('average best effort jct of scheduler %s is %02f, %02f (hour)'%(opt.schedule,  best_metric / best_number, best_metric / best_number / 3600))
            logger.info('weighted deadline miss rate of scheduler %s is %02f percent'%(opt.schedule, slo_metric / slo_number * 100))



def main(opt, logger):
    cluster_manager = prepare_cluster(opt)
    job_manager = prepare_job_manager() 
    user_manager = None 
    parse_job_file(filename=opt.trace, job_type=opt.job_type, job_manager=job_manager, opt=opt)
    
    global PM
    PM = PlaceMentFactory(cluster_manager=cluster_manager, name=opt.placement) # construct placement after init cluster

    if not os.path.exists(opt.save_log_dir):
        os.makedirs(opt.save_log_dir) 
    
    if opt.schedule == 'yarn-cs':
        scheduler = YarnCSScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager, placement=PM, name=opt.schedule, \
                                    logger=logger, scheduling_time_interval=opt.scheduling_time_interval, save_dir=opt.save_log_dir)
    elif opt.schedule == 'edf':
        scheduler = EDFScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager,  placement=PM, name=opt.schedule, \
                                        logger=logger, check_time_interval = opt.scheduling_time_interval, save_dir=opt.save_log_dir)
    elif opt.schedule == 'tetri-sched': 
        scheduler = TetriSchedScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager, placement=PM, name=opt.schedule, \
                                        logger=logger, scheduling_time_interval=opt.scheduling_time_interval, 
                                        lease_term_interval=opt.lease_term_interval,save_dir=opt.save_log_dir)
    elif opt.schedule == 'tiresias':
        num_queue = opt.num_queue
        queue_limit = [3600, 1000 * 3600]
        scheduler = TiresiasScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager, placement=PM, name=opt.schedule, \
                                        logger=logger, num_queue=num_queue, queue_limit=queue_limit, \
                                        solve_starvation=0, scheduling_time_interval = opt.scheduling_time_interval, save_dir=opt.save_log_dir)
    elif opt.schedule == 'themis':
        scheduler = ThemisScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager, placement=PM, name=opt.schedule, \
                                        logger=logger, scheduling_time_interval=opt.scheduling_time_interval, 
                                        lease_term_interval=opt.lease_term_interval,save_dir=opt.save_log_dir)
    elif opt.schedule == 'tetri-sched': 
        scheduler = TetriSchedScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager, placement=PM, name=opt.schedule, \
                                        logger=logger, scheduling_time_interval=opt.scheduling_time_interval, 
                                        lease_term_interval=opt.lease_term_interval,save_dir=opt.save_log_dir)
    elif opt.schedule == 'titan': 
        scheduler = TitanScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager, placement=PM, name=opt.schedule, \
                                    logger=logger, scheduling_time_interval = opt.scheduling_time_interval, save_dir=opt.save_log_dir, \
                                    multi_task_adaptivity=opt.multi_task_adaptivity, temporal_transferability=opt.temporal_transferability)
    elif opt.schedule == 'hpo_titan': 
        scheduler = HPOTitanScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager, placement=PM, name=opt.schedule, \
                                    logger=logger, scheduling_time_interval = opt.scheduling_time_interval, save_dir=opt.save_log_dir, \
                                    multi_task_adaptivity=opt.multi_task_adaptivity, temporal_transferability=opt.temporal_transferability)
    elif opt.schedule == 'optimus': 
        scheduler = OptimusScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager, placement=PM, name=opt.schedule, \
                                        logger=logger, scheduling_time_interval=opt.scheduling_time_interval, 
                                        lease_term_interval=opt.lease_term_interval,save_dir=opt.save_log_dir)
    elif opt.schedule == 'pollux': 
        scheduler = PolluxScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager, placement=PM, name=opt.schedule, \
                                        logger=logger, scheduling_time_interval=opt.scheduling_time_interval, 
                                        lease_term_interval=opt.lease_term_interval,save_dir=opt.save_log_dir)
    elif opt.schedule == 'sigma': 
        scheduler = SigmaScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager, placement=PM, name=opt.schedule, \
                                        logger=logger, scheduling_time_interval=opt.scheduling_time_interval, 
                                        lease_term_interval=opt.lease_term_interval,save_dir=opt.save_log_dir)
    elif opt.schedule == 'srtf':
        scheduler = ShortestRemainingTimeFirstScheduler(job_manager=job_manager, cluster_manager=cluster_manager, user_manager=user_manager, placement=PM, name=opt.schedule, \
                                    logger=logger, scheduling_time_interval = opt.scheduling_time_interval, save_dir=opt.save_log_dir)
    else:
        raise NotImplementedError

    scheduler.run()
    summary_all_jobs(job_manager=job_manager)
    logger.info(opt)



if __name__ == '__main__':

    opt = options.Singleton.init()
    if os.path.isdir(opt.trace):
        
        args_list = list() 
        opt_list = list() 
        for workload in glob.glob(opt.trace + "/*.csv"): 
            name = os.path.basename(workload)[:-4]
            logger = getLogger(name='log/{}_{}_{}_{}'.format(opt.schedule, opt.placement, opt.ident, name), level=opt.log_level)
            opt_list.append(copy.deepcopy(opt))
            opt_list[-1].trace = workload
            opt_list[-1].save_log_dir = opt.save_log_dir + "/" + name
            args_list.append((opt_list[-1], logger))

        with multiprocessing.Pool(processes=len(opt_list)) as pool:
            ret_list = pool.map(main, args_list)
        
    else: 
        if not os.path.exists('log/'):
            os.makedirs('log/')
        logger = getLogger(name='log/{}_{}_{}'.format(opt.schedule, opt.placement, opt.ident), level=opt.log_level)
        main(opt, logger)