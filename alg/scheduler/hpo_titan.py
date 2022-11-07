import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
import math 
import collections
import numpy as np 

from client.job.state import JobState
from .base import BaseScheduler
from .schedutils import resource_summary, schedule_summary, job_summary, hpo_schedule_summary
from .titan_solver import TitanSolver, TitanMultiTaskAdaptivitySolver
from .titan_mtask import mtask_builder
from .titan_transfer import transfer_builder, temporal_transfer_builder
from .titan_utils import compute_weight_metric, allocation2num, build_placement_from_num, \
                        create_candidate_allocations, METHOD, append, application_priority
from enum import Enum

CRITICAL_EPOCH_POINT = [1, 3, 9, 100]
MAX_JOB_PER_CLUSTER=[27, 9, 3, 1]
TRIAL_THRESHOLD = 10
MAX_RING = 4


class HpoState(Enum): 
    UNKNOWN = 1
    KILL = 2 
    SURVIVE = 3




def cal_bucket(idx, ring, reduction_factor): 
    for i in range(ring): 
        idx = idx / reduction_factor
    return idx 
        


class HPOTitanScheduler(BaseScheduler):
    def __init__(self, job_manager, cluster_manager, user_manager, placement, name, logger, **kwargs):
        super(HPOTitanScheduler, self).__init__(job_manager, cluster_manager, user_manager, placement=placement, name=name, logger=logger)
        assert self.name == 'hpo_titan'
        self.reduction_factor = 3

        self.pending_jobs = job_manager.pending_jobs
        self.running_jobs = job_manager.running_jobs
        self.event_jobs = job_manager.event_jobs
        self.completion_jobs = list()
        self.scheduling_time_interval = kwargs.get('scheduling_time_interval', 10)
        self.save_dir = kwargs.get('save_dir', 'result/')
        
        self.titan_solver = TitanSolver(method='naive')
        self.temporal_transferability =kwargs.get('temporal_transferability', False)
        self.solver_time_list = list() 
        self.hpo_job_clusters = dict() 
        self.completion_hpo_job_clusters = dict() 
        self.running_hpo_job_clusters = dict() 
        self.pending_hpo_job_clusters = dict() 
        self.hpo_job_clusters_by_epoch = dict()
        self.hpo_winners = list() 
        self.hpo_applications = list() 
        self.former_application = None 
        self.later_application = None
        self.heterogeneity = False
        self.records = list() 
        # [appname][hpo_param] = state
        self.hpo_states = dict() 
        # [appname][hpo_param] = idx
        self.hpo_idx = dict() 
        # [appname][ring][ring_idx] = metric
        self.metric_per_ring = dict() 
        self.metric_per_ring_to_job = dict() 
        self.metric_per_ring_to_state = dict() 
        self.bad_candidates = list() 
        self.good_candidates = list() 


    def clean_clusters(self, ): 
        self.hpo_job_clusters = dict() 
        self.completion_hpo_job_clusters = dict() 
        self.running_hpo_job_clusters = dict() 
        self.pending_hpo_job_clusters = dict() 
        self.hpo_job_clusters_by_epoch = dict() 


    def create_job_clusters(self, cur_time): 
        self.clean_clusters() 

        # for job in self.completion_jobs: 
        #     append(job, self.completion_hpo_job_clusters)
        #     append(job, self.hpo_job_clusters)

        # for job in self.running_jobs: 
        #     append(job, self.running_hpo_job_clusters)
        #     append(job, self.hpo_job_clusters)
        
        # for job in self.pending_jobs: 
        #     append(job, self.pending_hpo_job_clusters)
        #     append(job, self.hpo_job_clusters)

        if len(self.hpo_states) == 0 and len(self.hpo_idx) == 0:
            assert len(self.running_jobs) == 0
            for job in self.pending_jobs: 
                job.ring = 0 
                appname = job.application.name
                if appname not in self.hpo_states: 
                    self.hpo_states[appname] = dict() 

                job.hpo_param = (job.target_lr, job.target_gradient_steps)
                hpo_param = job.hpo_param
                if hpo_param not in self.hpo_states:
                    self.hpo_states[appname][hpo_param] = HpoState.UNKNOWN

            param2idx = dict() 
            for appname in self.hpo_states.keys(): 
                if appname not in self.hpo_idx: 
                    self.hpo_idx[appname] = dict() 
                for idx, hpo_param in enumerate(sorted(self.hpo_states[appname].keys())): 
                    if idx not in param2idx:  
                        param2idx[hpo_param] = idx
                    self.hpo_idx[appname][hpo_param] = param2idx[hpo_param]
            
            for job in self.pending_jobs: 
                hpo_param = job.hpo_param
                job.ring_idx = self.hpo_idx[job.application.name][hpo_param]
                job.hpo_idx =self.hpo_idx[job.application.name][hpo_param] # care about the difference between ring_idx and hpo_idx

            for appname in self.hpo_states.keys(): 
                self.metric_per_ring[appname] = dict()
                self.metric_per_ring_to_job[appname] = dict() 
                self.metric_per_ring_to_state[appname] = dict() 
                for ring in range(MAX_RING): 
                    length = MAX_JOB_PER_CLUSTER[ring]
                    self.metric_per_ring[appname][ring] = [-1 for _ in range(length)]
                    self.metric_per_ring_to_job[appname][ring] = [None for _ in range(length)]
                    self.metric_per_ring_to_state[appname][ring] = [HpoState.UNKNOWN for _ in range(length)]
            
            for job in self.pending_jobs: 
                appname = job.application.name 
                hpo_param = job.hpo_param 
                hpo_idx = self.hpo_idx[appname][hpo_param]
                self.metric_per_ring_to_job[appname][0][hpo_idx] = job


            assert len(self.hpo_states) == 2 
            assert len(self.hpo_idx) == 2 
            
        if len(self.hpo_applications) == 0: 
            self.hpo_applications = sorted(list(self.hpo_states.keys()), key=lambda app: application_priority(app))
        
        if self.former_application is None: 
            self.former_application = self.hpo_applications[0]

        if self.later_application is None: 
            self.later_application = self.hpo_applications[1]
        

        for job in self.running_jobs + self.pending_jobs: 
            pass
        return 
        # key step 
        # for appname in self.hpo_job_clusters.keys(): 
        #     # appname, e.g., wnli, snli, cifar10
        #     if appname not in self.hpo_job_clusters_by_epoch: 
        #         self.hpo_job_clusters_by_epoch[appname] = dict() 
            
        #     # for epoch in CRITICAL_EPOCH_POINT: 
        #     #     if epoch not in self.hpo_job_clusters_by_epoch[appname]: 
        #     #         self.hpo_job_clusters_by_epoch[appname][epoch] = list() 

        #     # assign metrics_per_ring
        #     need_remove_jobs = list() 
        #     for ring, epoch in enumerate(CRITICAL_EPOCH_POINT): 
        #         for job in self.hpo_job_clusters[appname]: 
        #             if job.get_current_epoch() >= epcoh: 
        #                 hpo_param = job.hpo_param 
        #                 idx = self.hpo_idx[appname][hpo_parm]
        #                 ring_idx = cal_bucket(idx, ring, self.reduction_factor)
        #                 if self.metric_per_ring[appname][ring][ring_idx] == -1: 
        #                     self.metric_per_ring_to_job[appname][ring][ring_idx] = job 
        #                     self.metric_per_ring[appname][ring][ring_idx] = job.query_metric(epoch)
        #                 elif self.metric_per_ring_to_job[appname][ring][ring_idx].name != job.name and job.completion is not None: 
        #                     job.early_stop(cur_time) 
        #                     job.status = JobState.END
        #                     self.completion_jobs.append(job)
        #                     need_remove_jobs.append(job)
        #                 job.ring_idx = ring_idx
        #                 job.ring = ring 
                        

        #     for job in need_remove_jobs: 
        #         if job in self.pending_jobs: 
        #             self.pending_jobs.remove(job)
        #         if job in self.running_jobs: 
        #             self.running_jobs.append(job)

    
    def debug_cluster(self, cur_time): 
        self.logger.info('event {}, pending {}, running {}, completion {}'.format(len(self.event_jobs), len(self.pending_jobs), len(self.running_jobs), len(self.completion_jobs)))

        # if len(self.pending_jobs) == 10 and len(self.running_jobs) == 0: 
        #     for job in self.pending_jobs: 
        #         hpo_param = (job.target_lr, job.target_gradient_steps)
        #         print(hpo_param in self.bad_candidates)
        #         self.logger.info(hpo_param)
        #         self.logger.info("    {}:\t[placement {}]\t[progress {:.2f}%]\t[rank {}]\t[max_peers {}]\t[epoch {}]".format(
        #               job.name, job.placement, job.progress / job.max_progress * 100, job.rank, min(job.number_of_peers, job.max_of_peers), job.get_current_epoch()))
        #         import pdb; pdb.set_trace()  

        # tot_jobs = self.event_jobs + self.pending_jobs + self.running_jobs + self.completion_jobs
        # if len(self.event_jobs) + len(self.pending_jobs) + len(self.running_jobs) + len(self.completion_jobs) != 54: 
        #     for job in self.job_manager.job_list: 
        #         if job not in tot_jobs: 
        #             import pdb; pdb.set_trace() 

    def finish_all_jobs(self, ): 
        return len(self.event_jobs) + len(self.pending_jobs) + len(self.running_jobs) == 0
    
    def release_job_resource(self, job, status=JobState.END):
        if self.placement.name == 'gandiva':
            ret = self.cluster_manager.release_gandiva_job_resource(job, status)
        else:
            ret = self.cluster_manager.release_job_resource(job, status)
        return ret
        

    def try_allocate_resoure(self, job):
        if self.placement.name in ['random', 'consolidate', 'gandiva', 'local_search']:
            if self.placement.__alias__ == 'meta': 
                ret = self.placement.place_jobs(job)
            else: 
                ret = self.placement.place_jobs(job)
        else:
            raise NotImplementedError
        return ret
        

    # abstract
    def place_jobs(self, jobs, cur_time):
        jobs = sorted(jobs, key=lambda e: -e.target_num_gpus)
        for job in jobs:
            if not self.try_allocate_resoure(job): 
                continue 
            self.logger.info('pending job {} is resumed at time {}, placement {}'.format(job.name, cur_time, job.placement))


    # abstract
    def check_resource(self, **kwargs):
        resource_name = kwargs.get('resource', 'gpu')
        assert isinstance(resource_name, str)
        if 'gpu' == resource_name:
            return self.cluster_manager.check_free_gpus()
        if 'cpu' == resource_name:
            return self.cluster_manager.check_free_cpus()
        raise NotImplementedError
    
    def execute_preempt(self, job, cur_time):
        assert self.release_job_resource(job, status=JobState.PENDING) == True
        job.release_resource() 
        self.logger.info('running job {} is preempted at time {}'.format(job.name, cur_time))

    def execute_start_job(self, start_job, cur_time):
        self.pending_jobs.append(start_job)
        start_job.status = JobState.PENDING
        self.logger.info('---- job[{}] is added  at time[{}]'.format(start_job.name, cur_time))


    def flush_event_jobs(self, prev_time, cur_time):
        event_list = list()
        for event_job in self.event_jobs:
            if event_job.submission_time <= cur_time:
                assert event_job.submission_time >= prev_time
                event_list.append(event_job)
        

        submit_job_num = 0
        for event_job in event_list: 
            self.execute_start_job(event_job, cur_time)
            self.event_jobs.remove(event_job)
            submit_job_num += 1

        self.submit_job_num_list.append(submit_job_num)
        

    def flush_running_jobs(self, prev_time, cur_time):
        need_remove_jobs = list()
        self.logger.info('the number of running jobs is {}'.format(len(self.running_jobs)))
        used_gpus = 0
        
        # self.debug_cluster(cur_time) 
        for job in self.running_jobs:
            job.step(cur_time - max(prev_time, job.submission_time))
            self.logger.info("    {}:\t[placement {}]\t[progress {:.2f}%]\t[rank {}]\t[max_peers {}]".format(
                      job.name, job.placement, job.progress / job.max_progress * 100, job.rank, min(job.number_of_peers, job.max_of_peers)))
            used_gpus += sum(job.placement)
            if job.completion_time is not None: 
                self.release_job_resource(job) == True
                if True: 
                    job.status = JobState.END
                    self.completion_jobs.append(job)
                    need_remove_jobs.append(job)
                    self.hpo_winners.append(job)
                    self.logger.info('running job {} is finished at time {}, duration is {}, epoch is {}'.format(job.name, cur_time, job.completion_time - job.submission_time, job.get_current_epoch()))

        self.logger.info("GPU utilization: {}".format(used_gpus))
        for job in need_remove_jobs:
            self.running_jobs.remove(job)

    def flush_pending_jobs(self, prev_time, cur_time):
        need_remove_jobs = list() 
        early_stop_jobs = list() 
        for job in self.pending_jobs:
            job.step(seconds=cur_time - max(prev_time, job.submission_time))

        #     continue 
        #     if not hasattr(job, 'hpo_param'): continue 
        #     appname = job.application.name 
        #     hpo_param = job.hpo_param 
        #     if self.satisfy_early_stop(job): 
        #         early_stop_jobs.append(job)
        #     else: 
        #         ring = job.ring 
        #         ring_idx = job.ring_idx 
        #         candidate_list = self.metric_per_ring[appname][ring]
        #         mymetric = candidate_list[idx]
        #         left = ring_idx // self.reduction_factor * self.reduction_factor
        #         right = ring_idx // self.reduction_factor * self.reduction_factor + self.reduction_factor
        #         candidate_list = candidate_list[left:right]

        #         if mymetric > -1 and mymetric < max(candidate_list): 
        #             job.early_stop(cur_time)
        #             job.status = JobState.END
        #             self.completion_jobs.append(job)
        #             need_remove_jobs.append(job)
        #             self.logger.info('pending job {} is early stop at time {}, duration is {}, epoch is {}'.format(job.name, cur_time, job.completion_time - job.submission_time, job.get_current_epoch()))
        #             # self.logger.info('number of peers {}, max_peers {}, rank {}, topk {}'.format(job.number_of_peers, job.max_of_peers, job.rank, topk))

        # for job in early_stop_jobs: 
        #     job.early_stop(cur_time)
        #     job.status = JobState.END
        #     self.completion_jobs.append(job)
        #     self.pending_jobs.remove(job)


        # for job in need_remove_jobs:
        #     self.pending_jobs.remove(job)
        #     if self.temporal_transferability and job.application.name == self.former_application: 
        #         ring = job.ring 
        #         ring_idx = job.ring_idx 
        #         if ring == 0: 
        #             self.hpo_states[job.application.name][job.hpo_param] = HPOState.KILL
        #             if hpo_param not in self.bad_candidates: 
        #                 self.bad_candidates.append(hpo_param)

            # if self.temporal_transferability and job.get_current_epoch() < CRITICAL_EPOCH_POINT[1] and job.application.name == self.former_application:
            #     hpo_param = (job.target_lr, job.target_gradient_steps) 
            #     if hpo_param not in self.bad_candidates: 
            #         self.bad_candidates.append(hpo_param)

            

    def normalized_weight(self, weight_list, power): 
        if power < 0: 
            if len(weight_list) > 0 and max(weight_list) > 10000: 
                normalized_weight = max(weight_list) / 100 
                weight_list = [weight / normalized_weight for weight in weight_list]
        else: 
            if len(weight_list) > 0 and min(weight_list) < 1e-2: 
                normalized_weight = min(weight_list) / 1e-2
                if normalized_weight == 0: 
                    import pdb; pdb.set_trace() 
                weight_list = [weight / normalized_weight for weight in weight_list]
        return weight_list


    def satisfy_early_stop(self, job): 
        if job.application.name == self.later_application and self.temporal_transferability: 
            if self.hpo_states[self.former_application][hpo_param] == HPOState.KILL: 
                return True 
        return False

    # def flush_job_history(self, job): 
    #     appname = job.application.name
    #     hpo_param = job.hpo_param
    #     hpo_idx = self.hpo_idx[appname][hpo_param]
    #     ring_idx = hpo_idx
    #     survive = True 
    #     for ring in range(job.ring): 
    #         ring_idx = ring_idx // self.reduction_factor 
        
    #     for ring in range(job.ring, MAX_RING): 
    #         critical_epoch = CRITICAL_EPOCH_POINT[ring]
    #         if job.get_current_epoch() >= critical_epoch:
    #             mymetric = job.query_metric(critical_epoch)

    #             if self.metric_per_ring[appname][ring][ring_idx] != -1: 
    #                 survive = False 
    #             else: 
    #                 self.metric_per_ring[appname][ring][ring_idx] = mymetric 
    #                 left = ring_idx // self.reduction_factor * self.reduction_factor 
    #                 right = ring_idx // self.reduction_factor * self.reduction_factor + self.reduction_factor
    #                 candidate_list = 
    #             ring_idx = ring_idx // self.reduction_factor 

    def early_stop(self, job, cur_time): 
        if job.status == JobState.RUNNING: 
            self.release_job_resource(job)
        job.early_stop(cur_time) 
        job.status = JobState.END
        self.completion_jobs.append(job)
        if job in self.running_jobs: 
            self.running_jobs.remove(job)
            
        if job in self.pending_jobs: 
            self.pending_jobs.remove(job)

        if job.application.name == 'vit@mnist': 
            if job.get_current_epoch() >= 3: 
                self.logger.info('job {} is early stop at time {}, duration is {}, epoch is {}'.format(job.name, cur_time, job.completion_time - job.submission_time, job.get_current_epoch()))

    def fast_ignore(self, job): 
        return self.temporal_transferability and job.application.name == self.later_application and (job.hpo_param not in self.bad_candidates) and (job.hpo_param not in self.good_candidates) 


    def flush_runnable_jobs(self, prev_time, cur_time):
        
        should_run_jobs = list() 
        early_stop_jobs = list() 
        need_preempt_jobs = list() 
        for ring in range(MAX_RING - 1): 
            critical_epoch = CRITICAL_EPOCH_POINT[ring]
            # update current metric_per_ring information 
            for job in self.running_jobs + self.pending_jobs + self.completion_jobs: 
                if self.fast_ignore(job): 
                    continue

                appname = job.application.name
                if job.get_current_epoch() >= critical_epoch and job.ring == ring: 
                    mymetric = job.query_metric(critical_epoch)
                    self.metric_per_ring[appname][ring][job.ring_idx] = mymetric
                    self.metric_per_ring_to_job[appname][ring][job.ring_idx] = job 
                    
                
            for appname in self.metric_per_ring.keys(): 
                metric_list = self.metric_per_ring[appname][ring]
                job_list = self.metric_per_ring_to_job[appname][ring]
                state_list = self.metric_per_ring_to_state[appname][ring]
                for left in range(0, len(state_list), 3): 
                    max_metric = max(metric_list[left:left+3])
                    min_metric = min(metric_list[left:left+3])
                    choice_list = list() 
                    for candidate_id in range(3): 
                        current_id = left + candidate_id
                        if metric_list[current_id] >= 0 and metric_list[current_id] < max_metric: 
                            state_list[current_id] = HpoState.KILL
                            early_stop_jobs.append(job_list[current_id])

                        elif metric_list[current_id] >= 0 and metric_list[current_id] == max_metric: 
                            choice_list.append(current_id)
                    

                    if min_metric >= 0: 
                        if max([job.ring for job in job_list[left:left+3]]) > ring: 
                            continue
                        assert len(choice_list) >= 1
                        candidate_id = choice_list[0]
                        choice_list.remove(candidate_id)
                        for current_id in choice_list: 
                            state_list[current_id] = HpoState.KILL
                            early_stop_jobs.append(job_list[current_id])
                        
                        state_list[candidate_id] = HpoState.SURVIVE 

                        job_list[candidate_id].ring += 1
                        if job_list[candidate_id].ring == 4: 
                            import pdb; pdb.set_trace() 
                            
                        job_list[candidate_id].ring_idx //= self.reduction_factor 
                        # import pdb; pdb.set_trace() 
                    else: 
                        for current_id in choice_list: 
                            need_preempt_jobs.append(job_list[current_id])
        
        if self.temporal_transferability: 
            self.bad_candidates = list() 
            self.good_candidates = list() 
            for state, job in zip(self.metric_per_ring_to_state[self.former_application][0], self.metric_per_ring_to_job[self.former_application][0]): 
                if state == HpoState.KILL: 
                    self.bad_candidates.append(job.hpo_param)
                if state == HpoState.SURVIVE: 
                    self.good_candidates.append(job.hpo_param)
                    

        for job in self.pending_jobs: 
            if self.temporal_transferability and job.application.name == self.later_application and job.hpo_param in self.bad_candidates: 
                self.metric_per_ring[job.application.name][0][job.ring_idx] = 1e-3 # the value is important
                early_stop_jobs.append(job)


        for job in early_stop_jobs: 
            # if job.name == 'roberta-base@snli-50': 
            #     import pdb; pdb.set_trace() 
            if job.completion_time is None: 
                self.early_stop(job, cur_time)
        

        for job in need_preempt_jobs: 
            if job in self.running_jobs: 
                self.running_jobs.remove(job)
                self.execute_preempt(job, cur_time)
                if job not in self.pending_jobs: 
                    self.pending_jobs.append(job)
        
        should_run_jobs = list() 
        for job in self.pending_jobs: 
            if self.fast_ignore(job): continue 
            appname = job.application.name 
            ring = job.ring 
            ring_idx = job.ring_idx 
            if self.metric_per_ring[appname][ring][ring_idx] < 0: 
                should_run_jobs.append(job)
        
        # self.logger.info('should run jobs {}'.format(len(should_run_jobs)))
        # if len(self.running_jobs) == 0 and len(self.pending_jobs) > 20 and len(should_run_jobs) == 0 and  cur_time > 1000: 
        #     print(self.metric_per_ring['roberta-base@mnli'][1])
        #     self.metric_per_ring_to_state['roberta-base@mnli'][0][6:9]
        #     for job in self.pending_jobs: 
        #         if job.application.name == 'roberta-base@mnli' and job.hpo_idx >= 6 and job.hpo_idx <= 8: 
        #             print('name {}, ring {}, ring_idx {}, {}'.format(job.name, job.ring, job.ring_idx, job.hpo_idx))
        #             import pdb; pdb.set_trace() 
       
        #     import pdb; pdb.set_trace() 

        # ////////////////////////////////////////// titan //////////////////////////////////////////
        # run jobs on free gpu resources 
        # a key step
        # runnable_jobs = self.running_jobs + should_run_jobs
        runnable_jobs = should_run_jobs
        
        if True: 
            required_resource_list = list() 
            weight_per_allocation_list = list() 
            equalivent_allocation_list = list() 

            total_gpu_num = self.cluster_manager.check_total_gpus() 
            runnable_jobs = sorted(runnable_jobs, key=lambda job: job.predict_remaining_time(1))
            if self.heterogeneity: 
                cluster_gpu_info = {
                    "A100": self.cluster_manager.check_total_gpus(key_info=["A100"]), 
                    "V100": self.cluster_manager.check_total_gpus(key_info=["V100"]),
                }
            else: 
                cluster_gpu_info = {"V100": self.cluster_manager.check_total_gpus()}

            candidate_allocations = create_candidate_allocations(self.cluster_manager, cluster_gpu_info, self.heterogeneity)

            if len(runnable_jobs) > total_gpu_num: 
                for job in runnable_jobs[total_gpu_num:]: 
                    if job.status == JobState.RUNNING: 
                        self.execute_preempt(job, cur_time)
                        if job in self.running_jobs: 
                            self.running_jobs.remove(job)
                        if job not in self.pending_jobs: 
                            self.pending_jobs.append(job)
                    if hasattr(job, 'equalivent_allocation_idx'):
                        delattr(job, 'equalivent_allocation_idx')
                runnable_jobs = runnable_jobs[:total_gpu_num]
                
            if len(runnable_jobs) > 0: 
                fair_placement = max(1, int(total_gpu_num / sum([job.job_number for job in runnable_jobs])))
                self.logger.info("fair_placement == {}".format(fair_placement))
            
            unique_job_num = len(runnable_jobs)
            for idx, job in enumerate(runnable_jobs):
                job.equalivent_allocation_idx = idx 
                fair_remaining_time = max(job.predict_remaining_time(min(fair_placement * job.job_number, job.max_num_gpus)), self.scheduling_time_interval)
                for allocations in candidate_allocations: 
                    if allocation2num(allocations) == 0: 
                        required_resource_list.append(allocations)
                        weight_per_allocation_list.append(1e-4*job.base_weight_scale)
                        equalivent_allocation_list.append(idx)
                        continue 
                    
                    if allocation2num(allocations) > job.max_num_gpus: continue 
                    # METHOD, job, placement, fair_placement, fair_remaining_time, cur_time, scheduling_time_interval
                    weight = compute_weight_metric(METHOD, job, allocations, fair_placement, fair_remaining_time, cur_time, self.scheduling_time_interval)
                    
                    # print(fair_placement, placement, weight, job.max_num_gpus)
                    # weight = 32. * gpu_num / step_time / (job.max_progress - job.progress) 
                    # print('max_progress', job.max_progress, job.progress, weight)
                    if np.isinf(weight) or np.isnan(weight): 
                        required_resource_list.append(allocations)
                        weight_per_allocation_list.append(1e-4*job.base_weight_scale)
                        equalivent_allocation_list.append(idx)
                        continue 
                    weight = weight # * job.reweight
                    required_resource_list.append(allocations)
                    weight_per_allocation_list.append(weight)
                    equalivent_allocation_list.append(idx)
        
        power = -1
        normalized_weight_per_allocation_list = self.normalized_weight(weight_per_allocation_list, power)
        weight_per_allocation_list = normalized_weight_per_allocation_list        
        if len(equalivent_allocation_list) == 0 and len(required_resource_list) > 0: 
            import pdb; pdb.set_trace() 
        solution = self.titan_solver.job_selection(required_resource_list, weight_per_allocation_list, \
                                        equalivent_allocation_list, unique_job_num, cluster_gpu_info, \
                                        max_seconds=30, power=power)
        # if unique_job_num == 1 and total_gpu_num > 4: 
        #     import pdb; pdb.set_trace() 
        self.logger.info('solution == {}'.format(solution))

        should_run_jobs = list() 
        for idx, job in enumerate(runnable_jobs): 
            solution[idx] = allocation2num(solution[idx])
            if job.status == JobState.RUNNING: 
                if job.target_num_gpus != solution[idx]: 
                    # if not hasattr(job, 'topology'): 
                    #     import pdb; pdb.set_trace() 

                    self.execute_preempt(job, cur_time)
                    if job in self.running_jobs: 
                        self.running_jobs.remove(job)
                    
                    if job not in self.pending_jobs: 
                        self.pending_jobs.append(job)

                if solution[idx] > 0: 
                    if job.target_num_gpus != solution[idx]: 
                        job.target_num_gpus = solution[idx] 
                        should_run_jobs.append(job)
                else: 
                    
                    job.target_num_gpus = None 
                    job.status = JobState.PENDING
                    if job not in self.pending_jobs: 
                        self.pending_jobs.append(job)
                    
            elif job.status == JobState.PENDING: 
                if solution[idx] > 0: 
                    job.target_num_gpus = solution[idx] 
                    should_run_jobs.append(job)

            else: 
                raise NotImplementedError 
        

        self.logger.info('free gpus {}'.format(self.cluster_manager.check_free_gpus() ))
        self.place_jobs(should_run_jobs, cur_time)
        
        for job in should_run_jobs: 
            if job.placement is not None: 
                if sum(job.placement) == 0: 
                    import pdb; pdb.set_trace() 
                if job in self.pending_jobs:
                    self.pending_jobs.remove(job)
                    job.status = JobState.PENDING
                if job not in self.running_jobs:
                    self.running_jobs.append(job)
                    job.status = JobState.RUNNING
            else: 
                # import pdb; pdb.set_trace() 
                job.target_num_gpus = None 

        self.logger.info('running jobs gpu allocations {}'.format([job.target_num_gpus for job in self.running_jobs]))
        self.logger.info('running jobs progress        {}'.format([job.max_progress - job.progress for job in self.running_jobs]))

                    

    def flush_jobs(self, prev_time, cur_time, status):
        if status == JobState.EVENT:
            self.flush_event_jobs(prev_time, cur_time)
        
        elif status == JobState.RUNNING:
            self.flush_running_jobs(prev_time, cur_time)
        
        elif status == JobState.PENDING:
            self.flush_pending_jobs(prev_time, cur_time)
        
        elif status == JobState.RUNNABLE:
            self.flush_runnable_jobs(prev_time, cur_time)

        else:
            raise NotImplementedError


    def run(self, ):
        cur_time = 0
        while not self.finish_all_jobs():
            # record resource statistics
            resource_summary(self)
            job_summary(self, cur_time)
            prev_time = max(0, cur_time - self.scheduling_time_interval)
            self.flush_jobs(prev_time, cur_time, status=JobState.RUNNING)
            self.flush_jobs(prev_time, cur_time, status=JobState.EVENT)
            self.flush_jobs(prev_time, cur_time, status=JobState.PENDING)
            self.create_job_clusters(cur_time) 
            self.flush_jobs(prev_time, cur_time, status=JobState.RUNNABLE)
            cur_time += self.scheduling_time_interval
            self.debug_cluster(cur_time)
            # if len(self.pending_jobs) == 1 and len(self.completion_jobs) == 53: 
            #     print(self.metric_per_ring["roberta-base@stsb"])
            #     for job in self.pending_jobs + self.running_jobs + self.completion_jobs: 
            #         if job.application.name == "roberta-base@stsb" and job.hpo_idx < 3: 
            #             print('name {}, epoch {}, ring {}, ring_idx {}, status {}'.format(job.name, job.get_current_epoch(), job.ring, job.ring_idx, job.status))
                    
            #     for job in self.pending_jobs: 
            #         # job = self.pending_jobs[-1]
            #         print('-'*10)
            #         print(job.name)
            #         print(job.ring)
            #         print(job.ring_idx)
            #         print(job.get_current_epoch())
            #         print(job.application.name)
            #     import pdb; pdb.set_trace() 

        for job in self.hpo_winners: 
            self.logger.info('job.name {}, lr = {}, graident_steps {}, metric {}'.format(job.name, job.target_lr, job.target_gradient_steps, job.query_metric(epoch=10)))
        schedule_summary(self)
        hpo_schedule_summary(self)
        