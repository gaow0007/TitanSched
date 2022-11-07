import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
import numpy as np 
from client.job.foundation_model import FoundationModelJob, TransferFoundationModelJob, TemporalTransferFoundationModelJob
from client.job.state import JobState
from .titan_utils import compute_weight_metric, allocation2num, build_placement_from_num, create_candidate_allocations, METHOD

TRANSFER_MAX_ALLOCATION_IDX=4096
TEMPORAL_TRANSFER_MAX_ALLOCATION_IDX=8192

def transfer_builder(self, runnable_jobs, prev_time, cur_time, required_resource_list, weight_per_allocation_list, equalivent_allocation_list):
    transfer_jobs = list() 
    number_of_transfer_training = 0 

    if len(self.pending_jobs) == 0: 
        return list(), list(), list(), list()

    intermediate_jobs = list() 
    intermediate_task_names = list() 
    for job in self.completion_jobs: 
        if job.application.task_name not in intermediate_task_names: 
            intermediate_task_names.append(job.application.task_name)
            intermediate_jobs.append(job)

    for job in self.pending_jobs:
        if job.progress == 0 and job.__alias__ == 'foundation_model' and hasattr(job, 'equalivent_allocation_idx') and (job in runnable_jobs): 
            transfer_jobs.append(job)
            number_of_transfer_training += 1
    max_equalivent_allocation_idx = max([job.equalivent_allocation_idx for job in runnable_jobs]) + TRANSFER_MAX_ALLOCATION_IDX
    

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


    transfer_required_resource_list = list() 
    transfer_weight_per_allication_list = list() 
    transfer_equalivent_allocation_list = list() 

    transfer_job_list = list() 
    for idxA in range(len(intermediate_jobs)): 
        jobA = intermediate_jobs[idxA]
        for idxB in range(number_of_transfer_training): 
            jobB = transfer_jobs[idxB]
            if jobA.application.task_name != jobB.application.task_name: 
                transfer_job = TransferFoundationModelJob(jobA, jobB)
                # if 'snli' in transfer_job.name: 
                #     self.logger.info('transfer weight {}, job name {}'.format(transfer_job.reweight, transfer_job.name))
                # import pdb; pdb.set_trace() 
                # self.logger.info('transfer weight {}, job name {}'.format(transfer_job.reweight, transfer_job.name))
                if transfer_job.reweight < 1.1: continue  
                fair_remaining_time = max(transfer_job.predict_remaining_time(min(fair_placement * transfer_job.job_number, transfer_job.max_num_gpus)), self.scheduling_time_interval)
                max_equalivent_allocation_idx += 1
                transfer_job_list.append(transfer_job)
                forbidden_job_id_list = [jobB.equalivent_allocation_idx, max_equalivent_allocation_idx]
                transfer_job.equalivent_allocation_idx = forbidden_job_id_list

                for allocations in candidate_allocations: 
                    if allocation2num(allocations) == 0: 
                        transfer_required_resource_list.append(allocations)
                        transfer_weight_per_allication_list.append(1e-4 * transfer_job.base_weight_scale)
                        transfer_equalivent_allocation_list.append(forbidden_job_id_list)
                        continue 
                    
                    if allocation2num(allocations) > job.max_num_gpus: continue 

                    
                    weight = compute_weight_metric(METHOD, job, allocations, fair_placement, fair_remaining_time, cur_time, self.scheduling_time_interval)
                    if np.isinf(weight) or np.isnan(weight): 
                        transfer_required_resource_list.append(allocations)
                        transfer_weight_per_allication_list.append(1e-4 * transfer_job.base_weight_scale)
                        transfer_equalivent_allocation_list.append(forbidden_job_id_list)
                        continue 
                    
                    weight = transfer_job.reweight * weight 
                    transfer_required_resource_list.append(allocations)
                    transfer_weight_per_allication_list.append(weight)
                    transfer_equalivent_allocation_list.append(forbidden_job_id_list)

    return transfer_required_resource_list, transfer_weight_per_allication_list, transfer_equalivent_allocation_list, transfer_job_list




def temporal_transfer_builder(self, runnable_jobs, prev_time, cur_time, required_resource_list, weight_per_allocation_list, equalivent_allocation_list):
    transfer_jobs = list() 
    number_of_transfer_training = 0 

    if len(self.pending_jobs) == 0: 
        return list(), list(), list(), list()
    
    for job in self.pending_jobs:
        if job.progress == 0 and job.__alias__ == 'foundation_model' and hasattr(job, 'equalivent_allocation_idx') and (job in runnable_jobs): 
            transfer_jobs.append(job)
            number_of_transfer_training += 1
    max_equalivent_allocation_idx = max([job.equalivent_allocation_idx for job in runnable_jobs]) + TEMPORAL_TRANSFER_MAX_ALLOCATION_IDX
    

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



    temporal_transfer_required_resource_list = list() 
    temporal_transfer_weight_per_allication_list = list() 
    temporal_transfer_equalivent_allocation_list = list() 

    temporal_transfer_job_list = list() 
    cnt = 0 
    for idxA in range(number_of_transfer_training): 
        jobA = transfer_jobs[idxA]
        for idxB in range(number_of_transfer_training): 
            jobB = transfer_jobs[idxB]
            if jobA.application.task_name != jobB.application.task_name: 
                temporal_transfer_job = TemporalTransferFoundationModelJob(jobA, jobB)
                # if 'snli' in temporal_transfer_job.name: 
                

                if temporal_transfer_job.reweight < 1.1: continue  
                # self.logger.info('temporal weight weight {}, job name {}'.format(temporal_transfer_job.reweight, temporal_transfer_job.name))
                fair_remaining_time = max(temporal_transfer_job.predict_remaining_time(min(fair_placement * temporal_transfer_job.job_number, temporal_transfer_job.max_num_gpus)), self.scheduling_time_interval)
                max_equalivent_allocation_idx += 1
                temporal_transfer_job_list.append(temporal_transfer_job)
                forbidden_job_id_list = [jobA.equalivent_allocation_idx, jobB.equalivent_allocation_idx]
                cnt += 1
                # self.logger.info('cnt == {}, forbidden_job_id_list {}, jobA.name {}, jobB.name {}'.format(cnt, forbidden_job_id_list, jobA.name, jobB.name))
                if len(candidate_allocations) == 0: 
                    import pdb; pdb.set_trace() 

                for allocations in candidate_allocations: 
                    if allocation2num(allocations) == 0: 
                        temporal_transfer_required_resource_list.append(allocations)
                        temporal_transfer_weight_per_allication_list.append(1e-4 * temporal_transfer_job.base_weight_scale)
                        temporal_transfer_equalivent_allocation_list.append(forbidden_job_id_list)
                        continue 
                    
                    if allocation2num(allocations) > job.max_num_gpus: continue 
                    
                    weight = compute_weight_metric(METHOD, job, allocations, fair_placement, fair_remaining_time, cur_time, self.scheduling_time_interval)

                    if np.isinf(weight) or np.isnan(weight): 
                        temporal_transfer_required_resource_list.append(allocations)
                        temporal_transfer_weight_per_allication_list.append(1e-4 * temporal_transfer_job.base_weight_scale)
                        temporal_transfer_equalivent_allocation_list.append(forbidden_job_id_list)
                        continue 
                    
                    weight = temporal_transfer_job.reweight * weight 
                    temporal_transfer_required_resource_list.append(allocations)
                    temporal_transfer_weight_per_allication_list.append(weight)
                    temporal_transfer_equalivent_allocation_list.append(forbidden_job_id_list)

    return temporal_transfer_required_resource_list, temporal_transfer_weight_per_allication_list, temporal_transfer_equalivent_allocation_list, temporal_transfer_job_list
