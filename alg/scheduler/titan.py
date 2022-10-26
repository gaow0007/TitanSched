import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
import numpy as np 
from client.job.foundation_model import FoundationModelJob, MergeFoundationModelJob
from client.job.state import JobState
from .base import BaseScheduler
from .schedutils import resource_summary, schedule_summary
from .titan_solver import TitanSolver, TitanMultiTaskAdaptivitySolver



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
        if self.multi_task_adaptivity: 
            self.titan_solver = TitanMultiTaskAdaptivitySolver(method='multi-task-adaptivity')
        else: 
            self.titan_solver = TitanSolver(method='naive')
        self.solver_time_list = list() 
    

    def debug_cluster(self, cur_time): 
        self.logger.info('event {}, pending {}, running {}, completion {}'.format(len(self.event_jobs), len(self.pending_jobs), len(self.running_jobs), len(self.completion_jobs)))
        tot_jobs = self.event_jobs + self.pending_jobs + self.running_jobs + self.completion_jobs
        if len(self.event_jobs) + len(self.pending_jobs) + len(self.running_jobs) + len(self.completion_jobs) != 160: 
            for job in self.job_manager.job_list: 
                if job not in tot_jobs: 
                    import pdb; pdb.set_trace() 

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
            self.logger.info('pending job {} is resumed at time {}'.format(job.name, cur_time))


    
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
            job.step(cur_time - max(prev_time, job.submission_time))
            self.logger.info("    {}:\t[placement {}]\t[progress {:.2f}%]".format(
                      job.name, job.placement, job.progress / job.max_progress * 100))
            used_gpus += sum(job.placement)
            if job.completion_time is not None: 
                self.release_job_resource(job) == True
                if isinstance(job, MergeFoundationModelJob): 
                    need_remove_jobs.append(job)
                    single_job_list = job.split_after_complete() 
                    for single_job in single_job_list: 
                        single_job.status = JobState.END
                        self.completion_jobs.append(single_job) 
                        self.logger.info('running job {} is finished at time {}'.format(single_job.name, cur_time))
                else: 
                    job.status = JobState.END
                    self.completion_jobs.append(job)
                    need_remove_jobs.append(job)
                    self.logger.info('running job {} is finished at time {}'.format(job.name, cur_time))

        self.logger.info("GPU utilization: {}".format(used_gpus))
        for job in need_remove_jobs:
            self.running_jobs.remove(job)

    def flush_pending_jobs(self, prev_time, cur_time):
        for job in self.pending_jobs:
            job.step(seconds=cur_time - max(prev_time, job.submission_time))



    def flush_runnable_jobs(self, prev_time, cur_time):
        runnable_jobs = self.pending_jobs + self.running_jobs
        unique_job_num = len(runnable_jobs)
        cluster_capacity = self.cluster_manager.check_total_gpus() 
        if cluster_capacity == 0: 
            return 

        required_resource_list = list() 
        weight_per_allocation_list = list() 
        equalivent_allocation_list = list() 
        if self.multi_task_adaptivity: 
            runnable_jobs = sorted(runnable_jobs, key=lambda job: (job.application.name, job.application.data_scale)) 

        total_gpu_num = self.cluster_manager.check_total_gpus() 
        max_allocated_gpu_num = 32
        # candidate_gpus = [i for i in range(0, total_gpu_num + 1)]
        candidate_gpus = [0, 1, 2, 3, 4] + [4 * i for i in range(2, total_gpu_num // 4 + 1)]
        if not self.multi_task_adaptivity: 
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
            fair_placement = max(1, int(total_gpu_num / len(runnable_jobs)))

        for idx, job in enumerate(runnable_jobs):

            fair_remaining_time = max(job.predict_remaining_time(min(fair_placement, job.max_num_gpus)), self.scheduling_time_interval)
            for gpu_num in candidate_gpus: 
                if gpu_num == 0: 
                    required_resource_list.append(gpu_num)
                    weight_per_allocation_list.append(1e-4)
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
                    weight_per_allocation_list.append(1e-4)
                    equalivent_allocation_list.append(idx)
                    continue 
                
                required_resource_list.append(gpu_num)
                weight_per_allocation_list.append(weight)
                equalivent_allocation_list.append(idx)
        

        if self.multi_task_adaptivity: 
            job_cluster_set = dict() 
            for idx, job in enumerate(self.pending_jobs): 
                if job.application.name not in job_cluster_set: 
                    job_cluster_set[job.application.name] = list() 
                job_cluster_set[job.application.name].append((idx, job))
            
            merge_required_resource_list = list() 
            merge_weight_per_allication_list = list() 
            merge_equalivent_allocation_list = list() 
            merge_unique_job_num = 0 
            merge_job_list = list() 

            for cluster_name, job_clusters in job_cluster_set.items(): 
                if len(job_clusters) > 1: 
                    forbidden_job_id_list = [job_info[0] for job_info in job_clusters]
                    merge_unique_job_num += 1
                    normalized_iteration_list = sorted([job_info[1].target_iteration for job_info in job_clusters])
                    reweight = sum(np.cumsum([item / normalized_iteration_list[0] for item in normalized_iteration_list])) / len(normalized_iteration_list)
                    self.logger.info("job len {}, reweight {}".format(len(forbidden_job_id_list), reweight)) 
                    merge_job = MergeFoundationModelJob([job_info[1] for job_info in job_clusters], cur_time) 
                    merge_job_list.append(merge_job)
                    self.logger.info('merge job iteration {}'.format(merge_job.target_iteration))
                    self.logger.info('single job iteration {}'.format([single_job.target_iteration for single_job in merge_job.fm_list]))
                    self.logger.info('comparison between merge {} and single {}'.format(merge_job.target_iteration * reweight, sum([single_job.target_iteration for single_job in merge_job.fm_list])))
                    for gpu_num in [1, 2, 4, 8]: 
                        merge_required_resource_list.append(gpu_num) 
                        merge_weight_per_allication_list.append(merge_job.application.get_max_throughput(placement=[gpu_num]) / merge_job.target_iteration * reweight)
                        merge_equalivent_allocation_list.append(forbidden_job_id_list)
            power = -1
            solution = self.titan_solver.job_selection(required_resource_list, weight_per_allocation_list, equalivent_allocation_list, unique_job_num, \
                                                    merge_required_resource_list, merge_weight_per_allication_list, merge_equalivent_allocation_list, \
                                                    merge_unique_job_num, cluster_capacity, max_seconds=5)
            should_run_jobs = list() 
            for idx, job in enumerate(self.pending_jobs): 
                if solution[idx] > 0: 
                    job.target_num_gpus = solution[idx]
                    should_run_jobs.append(job)

            should_run_merge_jobs = list() 
            for idx, merge_job in enumerate(merge_job_list): 
                if solution[idx + unique_job_num] > 0: 
                    merge_job.target_num_gpus = solution[idx + unique_job_num]
                    should_run_merge_jobs.append(merge_job)
            should_run_jobs = should_run_jobs + should_run_merge_jobs
            
        else:
            
            power = -1
            if power < 0: 
                if len(weight_per_allocation_list) > 0 and max(weight_per_allocation_list) > 10000: 
                    normalized_weight = max(weight_per_allocation_list) / 100 
                    weight_per_allocation_list = [weight / normalized_weight for weight in weight_per_allocation_list]
            else: 
                if len(weight_per_allocation_list) > 0 and min(weight_per_allocation_list) < 1e-2: 
                    normalized_weight = min(weight_per_allocation_list) / 1e-2
                    if normalized_weight == 0: 
                        import pdb; pdb.set_trace() 
                    weight_per_allocation_list = [weight / normalized_weight for weight in weight_per_allocation_list]
            
            # if len(self.running_jobs) + len(self.pending_jobs) > 10: 
            #     import pdb; pdb.set_trace()  
            
            solution = self.titan_solver.job_selection(required_resource_list, weight_per_allocation_list, \
                                                        equalivent_allocation_list, unique_job_num, cluster_capacity, \
                                                        max_seconds=30, power=power)
            # if len(self.running_jobs) + len(self.pending_jobs) > 30 and sum(solution) < 10: 
            #     import pdb; pdb.set_trace() 
            # power_list = [(weight_per_allocation_list[i] ** power) for i in range(len(weight_per_allocation_list))]
            self.logger.info('solution == {}'.format(solution))
            if len(self.pending_jobs) >= 20 and len(self.running_jobs) == 0 and sum(solution) == 0: 
                import pdb; pdb.set_trace() 
            should_run_jobs = list() 
            for idx, job in enumerate(runnable_jobs): 
                if job.status == JobState.RUNNING: 
                    if job.target_num_gpus != solution[idx]: 
                        if not hasattr(job, 'topology'): 
                            import pdb; pdb.set_trace() 

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
                if job not in self.pending_jobs: 
                    self.pending_jobs.append(job)
        # if cur_time == 6600: 
        #     for job in should_run_jobs: 
        #         if job.name == 'roberta-base@qnli-11': 
        #             import pdb; pdb.set_trace() 

        # self.debug_cluster(cur_time)
        self.logger.info('running jobs gpu allocations {}'.format([job.target_num_gpus for job in self.running_jobs]))
        self.logger.info('running jobs progress        {}'.format([job.max_progress - job.progress for job in self.running_jobs]))
        if self.multi_task_adaptivity: 
            for job in should_run_merge_jobs: 
                if job.placement is not None: 
                    for single_job in job.fm_list: 
                        self.pending_jobs.remove(single_job)
                        

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

            # record resource statistics
            resource_summary(self)
        
        schedule_summary(self)
        