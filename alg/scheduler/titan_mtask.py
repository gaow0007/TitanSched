import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
import numpy as np 
from client.job.foundation_model import FoundationModelJob, MtaskFoundationModelJob
from client.job.state import JobState


def mtask_builder(self, runnable_jobs, prev_time, cur_time, required_resource_list, weight_per_allocation_list, equalivent_allocation_list):
    mtask_jobs = list() 
    number_of_mtask_training = 0 

    if len(self.pending_jobs) == 0: 
        return list(), list(), list(), list()

    for job in self.pending_jobs:
        if job.progress == 0 and job.__alias__ == 'foundation_model' and hasattr(job, 'equalivent_allocation_idx'): 
            mtask_jobs.append(job)
            number_of_mtask_training += 1
    max_equalivent_allocation_idx = max([job.equalivent_allocation_idx for job in runnable_jobs])
    

    total_gpu_num = self.cluster_manager.check_total_gpus() 
    
    if len(runnable_jobs) > 0: 
        fair_placement = max(1, int(total_gpu_num / sum([job.job_number for job in runnable_jobs])))
    candidate_gpus = [0, 1, 2, 3, 4] + [4 * i for i in range(2, total_gpu_num // 4 + 1)]

    # mtask_required_resource_list.append(gpu_num) 
    # mtask_weight_per_allication_list.append(merge_job.application.get_max_throughput(placement=[gpu_num]) / merge_job.target_iteration * reweight)
    # mtask_equalivent_allocation_list.append(forbidden_job_id_list)
    mtask_required_resource_list = list() 
    mtask_weight_per_allication_list = list() 
    mtask_equalivent_allocation_list = list() 

    mtask_job_list = list() 
    for idxA in range(number_of_mtask_training): 
        jobA = mtask_jobs[idxA]
        for idxB in range(idxA + 1, number_of_mtask_training): 
            jobB = mtask_jobs[idxB]
            if jobA.application.task_name != jobB.application.task_name: 
                mtask_job = MtaskFoundationModelJob(jobA, jobB)
                # if 'snli' in mtask_job.name: 
                if mtask_job.reweight < 1.1: continue  
                self.logger.info('mtask weight {}, job name {}'.format(mtask_job.reweight, mtask_job.name))
                fair_remaining_time = max(mtask_job.predict_remaining_time(min(fair_placement * mtask_job.job_number, mtask_job.max_num_gpus)), self.scheduling_time_interval)
                max_equalivent_allocation_idx += 1
                mtask_job_list.append(mtask_job)
                forbidden_job_id_list = [jobA.equalivent_allocation_idx, jobB.equalivent_allocation_idx]
                mtask_job.equalivent_allocation_idx = forbidden_job_id_list

                for gpu_num in candidate_gpus: 
                    if gpu_num == 0: 
                        mtask_required_resource_list.append(gpu_num)
                        mtask_weight_per_allication_list.append(1e-4 * mtask_job.base_weight_scale)
                        mtask_equalivent_allocation_list.append(forbidden_job_id_list)
                        continue 
                    
                    if gpu_num > job.max_num_gpus: continue 
                    placement = () 
                    while sum(placement) < gpu_num:
                        placement = (*placement, min(gpu_num - sum(placement), 4))
                    from .titan import METHOD, compute_weight_metric
                    weight = compute_weight_metric(METHOD, job, placement, fair_placement, fair_remaining_time, cur_time, self.scheduling_time_interval)

                    if np.isinf(weight) or np.isnan(weight): 
                        mtask_required_resource_list.append(gpu_num)
                        mtask_weight_per_allication_list.append(1e-4 * mtask_job.base_weight_scale)
                        mtask_equalivent_allocation_list.append(forbidden_job_id_list)
                        continue 
                    
                    weight = mtask_job.reweight * weight 
                    mtask_required_resource_list.append(gpu_num)
                    mtask_weight_per_allication_list.append(weight)
                    mtask_equalivent_allocation_list.append(forbidden_job_id_list)

    return mtask_required_resource_list, mtask_weight_per_allication_list, mtask_equalivent_allocation_list, mtask_job_list

    #     power = -1
    #     solution = self.titan_solver.job_selection(required_resource_list, weight_per_allocation_list, equalivent_allocation_list, unique_job_num, \
    #                                             merge_required_resource_list, merge_weight_per_allication_list, merge_equalivent_allocation_list, \
    #                                             merge_unique_job_num, cluster_capacity, max_seconds=5)
    #     should_run_jobs = list() 
    #     for idx, job in enumerate(self.pending_jobs): 
    #         if solution[idx] > 0: 
    #             job.target_num_gpus = solution[idx]
    #             should_run_jobs.append(job)

    #     should_run_merge_jobs = list() 
    #     for idx, merge_job in enumerate(merge_job_list): 
    #         if solution[idx + unique_job_num] > 0: 
    #             merge_job.target_num_gpus = solution[idx + unique_job_num]
    #             should_run_merge_jobs.append(merge_job)
    #     should_run_jobs = should_run_jobs + should_run_merge_jobs
        
    # else:
        
    #     power = -1
    #     if power < 0: 
    #         if len(weight_per_allocation_list) > 0 and max(weight_per_allocation_list) > 10000: 
    #             normalized_weight = max(weight_per_allocation_list) / 100 
    #             weight_per_allocation_list = [weight / normalized_weight for weight in weight_per_allocation_list]
    #     else: 
    #         if len(weight_per_allocation_list) > 0 and min(weight_per_allocation_list) < 1e-2: 
    #             normalized_weight = min(weight_per_allocation_list) / 1e-2
    #             if normalized_weight == 0: 
    #                 import pdb; pdb.set_trace() 
    #             weight_per_allocation_list = [weight / normalized_weight for weight in weight_per_allocation_list]
        
    #     # if len(self.running_jobs) + len(self.pending_jobs) > 10: 
    #     #     import pdb; pdb.set_trace()  
        
    #     solution = self.titan_solver.job_selection(required_resource_list, weight_per_allocation_list, \
    #                                                 equalivent_allocation_list, unique_job_num, cluster_capacity, \
    #                                                 max_seconds=30, power=power)
    #     # if len(self.running_jobs) + len(self.pending_jobs) > 30 and sum(solution) < 10: 
    #     #     import pdb; pdb.set_trace() 
    #     # power_list = [(weight_per_allocation_list[i] ** power) for i in range(len(weight_per_allocation_list))]
    #     self.logger.info('solution == {}'.format(solution))
    #     if len(self.pending_jobs) >= 20 and len(self.running_jobs) == 0 and sum(solution) == 0: 
    #         import pdb; pdb.set_trace() 
    #     should_run_jobs = list() 
    #     for idx, job in enumerate(runnable_jobs): 
    #         if job.status == JobState.RUNNING: 
    #             if job.target_num_gpus != solution[idx]: 
    #                 if not hasattr(job, 'topology'): 
    #                     import pdb; pdb.set_trace() 

    #                 self.execute_preempt(job, cur_time)
    #                 if job in self.running_jobs: 
    #                     self.running_jobs.remove(job)
                    
    #                 if job not in self.pending_jobs: 
    #                     self.pending_jobs.append(job)

    #             if solution[idx] > 0: 
    #                 if job.target_num_gpus != solution[idx]: 
    #                     job.target_num_gpus = solution[idx] 
    #                     should_run_jobs.append(job)
    #             else: 
                    
    #                 job.target_num_gpus = None 
    #                 job.status = JobState.PENDING
    #                 if job not in self.pending_jobs: 
    #                     self.pending_jobs.append(job)
                    
    #         elif job.status == JobState.PENDING: 
    #             if solution[idx] > 0: 
    #                 job.target_num_gpus = solution[idx] 
    #                 should_run_jobs.append(job)

    #         else: 
    #             raise NotImplementedError 
    
    # self.logger.info('free gpus {}'.format(self.cluster_manager.check_free_gpus() ))
    # self.place_jobs(should_run_jobs, cur_time)
    
    # for job in should_run_jobs: 
    #     if job.placement is not None: 
    #         if sum(job.placement) == 0: 
    #             import pdb; pdb.set_trace() 
    #         if job in self.pending_jobs:
    #             self.pending_jobs.remove(job)
    #             job.status = JobState.PENDING
    #         if job not in self.running_jobs:
    #             self.running_jobs.append(job)
    #             job.status = JobState.RUNNING
    #     else: 
    #         # import pdb; pdb.set_trace() 
    #         job.target_num_gpus = None 
    #         if job not in self.pending_jobs: 
    #             self.pending_jobs.append(job)
    # # if cur_time == 6600: 
    # #     for job in should_run_jobs: 
    # #         if job.name == 'roberta-base@qnli-11': 
    # #             import pdb; pdb.set_trace() 

    # # self.debug_cluster(cur_time)
    # self.logger.info('running jobs gpu allocations {}'.format([job.target_num_gpus for job in self.running_jobs]))
    # self.logger.info('running jobs progress        {}'.format([job.max_progress - job.progress for job in self.running_jobs]))
    # if self.multi_task_adaptivity: 
    #     for job in should_run_merge_jobs: 
    #         if job.placement is not None: 
    #             for single_job in job.fm_list: 
    #                 self.pending_jobs.remove(single_job)