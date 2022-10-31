import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
import numpy as np 
from client.job.foundation_model import FoundationModelJob, MtaskFoundationModelJob, TemporalTransferFoundationModelJob, TransferFoundationModelJob
from client.job.state import JobState
from .base import BaseScheduler
from .schedutils import resource_summary, schedule_summary
from .titan_solver import TitanSolver, TitanMultiTaskAdaptivitySolver
from .titan_mtask import mtask_builder
from .titan_transfer import transfer_builder, temporal_transfer_builder


class TitanScheduler(BaseScheduler):
    def __init__(self, job_manager, cluster_manager, user_manager, placement, name, logger, **kwargs):
        super(TitanScheduler, self).__init__(job_manager, cluster_manager, user_manager, placement=placement, name=name, logger=logger)
        assert self.name == 'titan'
        self.pending_jobs = job_manager.pending_jobs
        self.running_jobs = job_manager.running_jobs
        self.event_jobs = job_manager.event_jobs
        self.completion_jobs = list()
        self.scheduling_time_interval = kwargs.get('scheduling_time_interval', 10)
        self.save_dir = kwargs.get('save_dir', 'result/')
        self.multi_task_adaptivity = kwargs.get('multi_task_adaptivity', False)
        if self.multi_task_adaptivity: # 0.391540
            self.titan_solver = TitanMultiTaskAdaptivitySolver(method='multi-task-adaptivity', logger=self.logger)
        else: # 0.391540
            self.titan_solver = TitanSolver(method='naive')
        self.temporal_transferability = (self.multi_task_adaptivity and kwargs.get('temporal_transferability', False))
        self.solver_time_list = list() 
    

    def debug_cluster(self, cur_time): 
        self.logger.info('event {}, pending {}, running {}, completion {}'.format(len(self.event_jobs), len(self.pending_jobs), len(self.running_jobs), len(self.completion_jobs)))
        # tot_jobs = self.event_jobs + self.pending_jobs + self.running_jobs + self.completion_jobs
        # if len(self.event_jobs) + len(self.pending_jobs) + len(self.running_jobs) + len(self.completion_jobs) != 160: 
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
        for job in self.running_jobs: 
            if isinstance(job, MtaskFoundationModelJob): 
                self.logger.info('mtask name {}, progress {}, max progress {}'.format(job.name, job.progress, job.max_progress))

        # self.debug_cluster(cur_time) 
        for job in self.running_jobs:
            job.step(cur_time - max(prev_time, job.submission_time))
            self.logger.info("    {}:\t[placement {}]\t[progress {:.2f}%]".format(
                      job.name, job.placement, job.progress / job.max_progress * 100))
            used_gpus += sum(job.placement)
            if job.completion_time is not None: 
                self.release_job_resource(job) == True
                if isinstance(job, MtaskFoundationModelJob): 
                    need_remove_jobs.append(job)
                    single_job_list = job.split_after_complete() 
                    self.logger.info('mtask running job {} is finished at time {}'.format(job.name, cur_time))
                    for single_job in single_job_list: 
                        # if single_job.name == 'vit-large@imagenet-split-2-22': 
                        #     import pdb; pdb.set_trace() 
                        single_job.status = JobState.END
                        self.completion_jobs.append(single_job) 
                        self.logger.info('running job {} is finished at time {}, duration is {}'.format(single_job.name, cur_time, single_job.completion_time - single_job.submission_time))
                elif isinstance(job, TransferFoundationModelJob): 
                    need_remove_jobs.append(job)
                    self.logger.info('single_transfer running job {} is finished at time {}'.format(job.name, cur_time))
                    single_job = job.split_after_complete() 
                    single_job.status = JobState.END
                    self.completion_jobs.append(single_job) 
                elif isinstance(job, TemporalTransferFoundationModelJob): 
                    need_remove_jobs(job)
                    single_job_list = job.split_after_complete() 
                    for single_job in single_job_list: 
                        single_job.status = JobState.END
                        self.completion_jobs.append(single_job) 
                        self.logger.info('running job {} is finished at time {}, duration is {}'.format(single_job.name, cur_time, single_job.completion_time - single_job.submission_time))
                else: 
                    job.status = JobState.END
                    self.completion_jobs.append(job)
                    need_remove_jobs.append(job)
                    self.logger.info('running job {} is finished at time {}, duration is {}'.format(job.name, cur_time, job.completion_time - job.submission_time))

        self.logger.info("GPU utilization: {}".format(used_gpus))
        for job in need_remove_jobs:
            self.running_jobs.remove(job)

    def flush_pending_jobs(self, prev_time, cur_time):
        for job in self.pending_jobs:
            job.step(seconds=cur_time - max(prev_time, job.submission_time))


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


    def flush_runnable_jobs(self, prev_time, cur_time):
        runnable_jobs = self.pending_jobs + self.running_jobs
        unique_job_num = len(runnable_jobs)
        cluster_capacity = self.cluster_manager.check_total_gpus() 
        if cluster_capacity == 0: 
            return 

        required_resource_list = list() 
        weight_per_allocation_list = list() 
        equalivent_allocation_list = list() 

        total_gpu_num = self.cluster_manager.check_total_gpus() 
        candidate_gpus = [i for i in range(0, total_gpu_num + 1)]
        # candidate_gpus = [0, 1, 2, 3, 4] + [4 * i for i in range(2, total_gpu_num // 4 + 1)]
        runnable_jobs = sorted(runnable_jobs, key=lambda job: job.predict_remaining_time(1))
        if len(runnable_jobs) > total_gpu_num: 
            for job in runnable_jobs[total_gpu_num:]: 
                if job.status == JobState.RUNNING: 
                    self.execute_preempt(job, cur_time)
                    if job in self.running_jobs: 
                        self.running_jobs.remove(job)
                    if job not in self.pending_jobs: 
                        self.pending_jobs.append(job)
            runnable_jobs = runnable_jobs[:total_gpu_num]
            
        if len(runnable_jobs) > 0: 
            fair_placement = max(1, int(total_gpu_num / sum([job.job_number for job in runnable_jobs])))
            self.logger.info("fair_placement == {}".format(fair_placement))

        for idx, job in enumerate(runnable_jobs):
            job.equalivent_allocation_idx = idx 
            fair_remaining_time = max(job.predict_remaining_time(min(fair_placement * job.job_number, job.max_num_gpus)), self.scheduling_time_interval)
            for gpu_num in candidate_gpus: 
                if gpu_num == 0: 
                    required_resource_list.append(gpu_num)
                    weight_per_allocation_list.append(1e-4*job.base_weight_scale)
                    equalivent_allocation_list.append(idx)
                    continue 
                
                if gpu_num > job.max_num_gpus: continue 
                placement = () 
                while sum(placement) < gpu_num:
                    placement = (*placement, min(gpu_num - sum(placement), 4))
                
                METHOD = "FAIR"
                if METHOD == "TIME":
                    predict_remaing_time = job.predict_remaining_time(placement)
                    # weight = fair_remaining_time / predict_remaing_time
                    weight = 1.0 / (predict_remaing_time + 1e-3) # * gpu_num
                elif METHOD == "THR": 
                    throughput = job.throughput_estimation(placement, batch_size=job.batch_size)
                    # step_time = job.application.get_throughput(placement=placement, local_bsz=32)
                    if isinstance(step_time, (tuple, np.ndarray)): 
                        step_time = step_time[0]
                    weight = 32. * gpu_num / step_time / (job.max_progress - job.progress)
                elif METHOD == "FAIR": 
                    predict_remaing_time = max(job.predict_remaining_time(placement), self.scheduling_time_interval)
                    weight = 1.0 * fair_remaining_time / predict_remaing_time 
                    print("predict_remaing_time == {}, {}, weight {}".format(fair_remaining_time, predict_remaing_time, weight))

                # print(fair_placement, placement, weight, job.max_num_gpus)
                # weight = 32. * gpu_num / step_time / (job.max_progress - job.progress) 
                # print('max_progress', job.max_progress, job.progress, weight)
                if np.isinf(weight) or np.isnan(weight): 
                    required_resource_list.append(gpu_num)
                    weight_per_allocation_list.append(1e-4*job.base_weight_scale)
                    equalivent_allocation_list.append(idx)
                    continue 
                weight = weight # * job.reweight
                required_resource_list.append(gpu_num)
                weight_per_allocation_list.append(weight)
                equalivent_allocation_list.append(idx)
                
        
        if self.multi_task_adaptivity: 
            mtask_required_resource_list, mtask_weight_per_allication_list, mtask_equalivent_allocation_list, mtask_jobs = \
                mtask_builder(self=self, runnable_jobs=runnable_jobs, prev_time=prev_time, cur_time=cur_time, required_resource_list=required_resource_list, \
                    weight_per_allocation_list=weight_per_allocation_list, equalivent_allocation_list=equalivent_allocation_list)
            
            transfer_required_resource_list, transfer_weight_per_allication_list, \
                transfer_equalivent_allocation_list, transfer_jobs = \
                list(), list(), list(), list() 
            temporal_transfer_required_resource_list, temporal_transfer_weight_per_allication_list, \
                temporal_transfer_equalivent_allocation_list, temporal_transfer_jobs = \
                list(), list(), list(), list() 
            if self.temporal_transferability: 
                temporal_transfer_required_resource_list, temporal_transfer_weight_per_allication_list, \
                    temporal_transfer_equalivent_allocation_list, temporal_transfer_jobs = temporal_transfer_builder(self=self, runnable_jobs=runnable_jobs, prev_time=prev_time, cur_time=cur_time, required_resource_list=required_resource_list, \
                        weight_per_allocation_list=weight_per_allocation_list, equalivent_allocation_list=equalivent_allocation_list)
            else: 
                transfer_required_resource_list, transfer_weight_per_allication_list, transfer_equalivent_allocation_list, transfer_jobs = \
                    transfer_builder(self=self, runnable_jobs=runnable_jobs, prev_time=prev_time, cur_time=cur_time, required_resource_list=required_resource_list, \
                        weight_per_allocation_list=weight_per_allocation_list, equalivent_allocation_list=equalivent_allocation_list)


        power = -1
        if self.multi_task_adaptivity: 
            tot_weight_weight_per_allocation_list = mtask_weight_per_allication_list + transfer_weight_per_allication_list + temporal_transfer_weight_per_allication_list
        else: 
            tot_weight_weight_per_allocation_list = list() 

        normalized_weight_per_allocation_list = self.normalized_weight(weight_per_allocation_list + (tot_weight_weight_per_allocation_list if self.multi_task_adaptivity else []), power)
        if self.multi_task_adaptivity: 
            # print(len(normalized_weight_per_allocation_list), len(weight_per_allocation_list), len(mtask_weight_per_allication_list))
            weight_per_allocation_list = normalized_weight_per_allocation_list[:len(weight_per_allocation_list)]
            assert len(mtask_weight_per_allication_list) == len(normalized_weight_per_allocation_list[len(weight_per_allocation_list):len(weight_per_allocation_list)+len(mtask_weight_per_allication_list)])
            mtask_weight_per_allication_list = normalized_weight_per_allocation_list[len(weight_per_allocation_list):len(weight_per_allocation_list)+len(mtask_weight_per_allication_list)]
            if self.temporal_transferability:
                temporal_transfer_weight_per_allication_list = normalized_weight_per_allocation_list[len(weight_per_allocation_list)+len(mtask_weight_per_allication_list):]
            else:  
                transfer_weight_per_allication_list = normalized_weight_per_allocation_list[len(weight_per_allocation_list)+len(mtask_weight_per_allication_list):]
            
        else: 
            weight_per_allocation_list = normalized_weight_per_allocation_list

        if self.multi_task_adaptivity: 
            unique_job_num = len(runnable_jobs)
            mtask_unique_job_num = len(mtask_jobs)
            transfer_unique_job_num = len(transfer_jobs)
            temporal_transfer_unique_job_num = len(temporal_transfer_jobs)
            # print([job.name for job in mtask_jobs])
            solution = self.titan_solver.job_selection(required_resource_list=required_resource_list, 
                                                    weight_per_allocation_list=weight_per_allocation_list, 
                                                    equalivent_allocation_list=equalivent_allocation_list, 
                                                    unique_job_num=unique_job_num, \
                                                    mtask_required_resource_list=mtask_required_resource_list, \
                                                    mtask_weight_per_allication_list=mtask_weight_per_allication_list, \
                                                    mtask_equalivent_allocation_list=mtask_equalivent_allocation_list, \
                                                    mtask_unique_job_num=mtask_unique_job_num, 
                                                    transfer_required_resource_list=transfer_required_resource_list, \
                                                    transfer_weight_per_allication_list=transfer_weight_per_allication_list, \
                                                    transfer_equalivent_allocation_list=transfer_equalivent_allocation_list, \
                                                    transfer_unique_job_num=transfer_unique_job_num, 
                                                    temporal_transfer_required_resource_list=temporal_transfer_required_resource_list, \
                                                    temporal_transfer_weight_per_allication_list=temporal_transfer_weight_per_allication_list, \
                                                    temporal_transfer_equalivent_allocation_list=temporal_transfer_equalivent_allocation_list, \
                                                    temporal_transfer_unique_job_num=temporal_transfer_unique_job_num, 
                                                    cluster_capacity=cluster_capacity, 
                                                    max_seconds=30, power=power)
        else: 
            solution = self.titan_solver.job_selection(required_resource_list, weight_per_allocation_list, \
                                            equalivent_allocation_list, unique_job_num, cluster_capacity, \
                                            max_seconds=30, power=power)

        self.logger.info('solution == {}'.format(solution))

        should_run_jobs = list() 
        for idx, job in enumerate(runnable_jobs): 
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
        
        if self.multi_task_adaptivity:
            for idx, mtask_job in enumerate(mtask_jobs): 
                if solution[unique_job_num + idx] > 0:
                    should_run_jobs.append(mtask_job)
                    mtask_job.target_num_gpus = solution[unique_job_num + idx]

            if self.temporal_transferability: 
                for idx, temporal_transfer_job in enumerate(temporal_transfer_jobs): 
                    if solution[unique_job_num + mtask_unique_job_num + idx] > 0: 
                        should_run_jobs.append(temporal_transfer_job)
                        temporal_transfer_job.target_num_gpus = solution[unique_job_num + mtask_unique_job_num + idx]
            else: 
                for idx, transfer_job in enumerate(transfer_jobs): 
                    if solution[unique_job_num + mtask_unique_job_num + idx] > 0: 
                        should_run_jobs.append(transfer_job)
                        transfer_job.target_num_gpus = solution[unique_job_num + mtask_unique_job_num + idx]

            
                


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
                if job not in self.pending_jobs and job not in mtask_jobs: 
                    self.pending_jobs.append(job)
        
        self.logger.info('running jobs gpu allocations {}'.format([job.target_num_gpus for job in self.running_jobs]))
        self.logger.info('running jobs progress        {}'.format([job.max_progress - job.progress for job in self.running_jobs]))

        # for job in self.pending_jobs: 
        #     if cur_time == 900: 
        #         if job.name == 'roberta-large@mnli-7': 
        #             import pdb; pdb.set_trace() 

        if self.multi_task_adaptivity:
            for job in mtask_jobs: 
                print('job name {}, job placement {}'.format(job.name, job.placement))
                if job.placement is not None:
                    self.logger.info('remove job A {}, job B {}'.format(job.modelA.name, job.modelB.name))
                    self.logger.info('mtask job equalivent_allocation_idx {}'.format(job.equalivent_allocation_idx))
                    self.pending_jobs.remove(job.modelA)
                    self.pending_jobs.remove(job.modelB)
            
            for job in transfer_jobs: 
                if job.placement is not None: 
                    
                    self.logger.info('transfer job A {}, job B {}'.format(job.modelA.name, job.modelB.name))
                    self.logger.info('transfer job equalivent_allocation_idx {}'.format(job.equalivent_allocation_idx))
                    if job.modelB not in self.pending_jobs: 
                        import pdb; pdb.set_trace() 
                    self.pending_jobs.remove(job.modelB) 

            for job in temporal_transfer_jobs: 
                if job.placement is not None: 
                    self.logger.info('transfer job A {}, job B {}'.format(job.modelA.name, job.modelB.name))
                    self.logger.info('transfer job equalivent_allocation_idx {}'.format(job.equalivent_allocation_idx))
                    if job.modelB not in self.pending_jobs: 
                        import pdb; pdb.set_trace() 
                    self.pending_jobs.remove(job.modelA)
                    self.pending_jobs.remove(job.modelB) 
                    

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
            prev_time = max(0, cur_time - self.scheduling_time_interval)
            self.flush_jobs(prev_time, cur_time, status=JobState.RUNNING)
            self.flush_jobs(prev_time, cur_time, status=JobState.EVENT)
            self.flush_jobs(prev_time, cur_time, status=JobState.PENDING)
            self.flush_jobs(prev_time, cur_time, status=JobState.RUNNABLE)
            cur_time += self.scheduling_time_interval
            self.debug_cluster(cur_time)
            # record resource statistics
            resource_summary(self)
        
        schedule_summary(self)
        