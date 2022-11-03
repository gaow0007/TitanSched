import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
import numpy as np 
from client.job.foundation_model import FoundationModelJob, MtaskFoundationModelJob
from client.job.state import JobState
from .titan_utils import compute_weight_metric, allocation2num, build_placement_from_num, create_candidate_allocations, METHOD

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


    if self.heterogeneity: 
        cluster_gpu_info = {
            "A100": self.cluster_manager.check_total_gpus(key_info=["A100"]), 
            "V100": self.cluster_manager.check_total_gpus(key_info=["V100"]),
        }
    else: 
        cluster_gpu_info = {"GPU": self.cluster_manager.check_total_gpus()}

    candidate_allocations = create_candidate_allocations(self.cluster_manager, cluster_gpu_info, self.heterogeneity)

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

                for allocations in candidate_allocations: 
                    if allocation2num(allocations) == 0: 
                        mtask_required_resource_list.append(allocations)
                        mtask_weight_per_allication_list.append(1e-4 * mtask_job.base_weight_scale)
                        mtask_equalivent_allocation_list.append(forbidden_job_id_list)
                        continue 
                    
                    if allocation2num(allocations) > job.max_num_gpus: continue 
                    placement = () 
                    weight = compute_weight_metric(METHOD, job, placement, fair_placement, fair_remaining_time, cur_time, self.scheduling_time_interval)

                    if np.isinf(weight) or np.isnan(weight): 
                        mtask_required_resource_list.append(allocations)
                        mtask_weight_per_allication_list.append(1e-4 * mtask_job.base_weight_scale)
                        mtask_equalivent_allocation_list.append(forbidden_job_id_list)
                        continue 
                    
                    weight = mtask_job.reweight * weight 
                    mtask_required_resource_list.append(allocations)
                    mtask_weight_per_allication_list.append(weight)
                    mtask_equalivent_allocation_list.append(forbidden_job_id_list)

    return mtask_required_resource_list, mtask_weight_per_allication_list, mtask_equalivent_allocation_list, mtask_job_list
