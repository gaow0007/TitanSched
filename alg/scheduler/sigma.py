import os, sys
import math
import copy
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from client.job.state import JobState, DDLState
from .base import BaseScheduler 
from .chronus_solver import SemiPreemptMIPSolver, compute_maximum_lease
from .schedutils import resource_summary, schedule_summary, MAX_LEASE_NUM, BE_REWARD, SLO_REWARD, MAX_SEARCH_JOB

def convert_to_bin(gpu_num):
    # if gpu_num == 3:
    #     return 4
    # elif gpu_num in [5, 6, 7]:
    #     return 8
    return gpu_num


# buggy code
class SigmaScheduler(BaseScheduler):
    def __init__(self, job_manager, cluster_manager, user_manager, placement, name, logger, **kwargs):
        super(SigmaScheduler, self).__init__(job_manager, cluster_manager, user_manager, placement=placement, name=name, logger=logger)
        assert self.name == 'sigma'
        self.pending_jobs = job_manager.pending_jobs
        self.running_jobs = job_manager.running_jobs
        self.event_jobs = job_manager.event_jobs
        self.completion_jobs = list()
        self.scheduling_time_interval = kwargs.get('scheduling_time_interval', 10)
        self.lease_term_interval = kwargs.get('lease_term_interval', 900)
        self.save_dir = kwargs.get('save_dir', 'result/')
        
        # lease related 
        self.cur_lease_index = 0
        self.next_lease_time = self.lease_term_interval
        total_resource_num = self.cluster_manager.check_total_gpus()
        self.resource_by_lease = [total_resource_num for _ in range(MAX_LEASE_NUM)]
        # mip_solver
        self.mip_solver = SemiPreemptMIPSolver('greedy')


    # abstract
    def check_resource(self, **kwargs):
        resource_name = kwargs.get('resource', 'gpu')
        assert isinstance(resource_name, str)
        if 'gpu' == resource_name:
            return self.CLUSTER.check_free_gpus()
        if 'cpu' == resource_name:
            return self.CLUSTER.check_free_cpus()
        
        raise NotImplementedError
        

    # abstract
    def place_jobs(self, jobs, cur_time):
        jobs = sorted(jobs, key=lambda e: -e.target_num_gpus)
        for job in jobs:
            if not self.try_allocate_resoure(job): 
                continue 
            self.logger.info('pending job {} is resumed at time {}'.format(job.name, cur_time))
            

    def release_job_resource(self, job, status=JobState.END):
        if self.placement.name == 'gandiva':
            ret = self.cluster_manager.release_gandiva_job_resource(job, status)
        else:
            ret = self.cluster_manager.release_job_resource(job, status)
        job.release_resource()
        return ret

    
    # allocate resource 
    def try_allocate_resoure(self, job):
        if self.placement.name in ['random', 'consolidate', 'gandiva', 'local_search', 'consolidate_random']:
            ret = self.placement.place_jobs(job)
        else:
            raise NotImplementedError
        return ret

    def execute_start_job(self, start_job, cur_time):
        self.pending_jobs.append(start_job)
        start_job.status = JobState.PENDING
        self.logger.info('---- job[{}] is added  at time[{}]'.format(start_job.name, cur_time))

    def execute_preempt(self, job, cur_time):
        assert self.release_job_resource(job, status=JobState.PENDING) == True
        self.logger.info('running job {} is preempted at time {}'.format(job.name, cur_time))

    def flush_running_jobs(self, prev_time, cur_time):
        need_remove_jobs = list()
        for job in self.running_jobs: 
            job.step(cur_time - max(prev_time, job.submission_time)) 
            # if self.is_best_effort(job): 
            #     job.completion_time = sys.maxsize # cur_time 
            #     assert self.release_job_resource(job) == True 
            #     job.status = JobState.END
            #     self.completion_jobs.append(job)
            #     need_remove_jobs.append(job)
            #     self.completion_jobs.append(job)
            #     continue 

            if job.completion_time is not None: 
                assert self.release_job_resource(job) == True 
                job.status = JobState.END
                self.completion_jobs.append(job)
                need_remove_jobs.append(job)
                self.completion_jobs.append(job)
            elif self.miss_ddl_job(job, cur_time): 
                self.execute_preempt(job, cur_time) 
                self.pending_jobs.append(job) # TODO
                need_remove_jobs.append(job)
                job.ddl_type = DDLState.BEST 
                job.cache_solution = None 
                            
        for job in need_remove_jobs:
            self.running_jobs.remove(job)
            

    def flush_pending_jobs(self, prev_time, cur_time): 
        for job in self.pending_jobs:
            job.step(seconds=cur_time - max(prev_time, job.submission_time))
            if self.miss_ddl_job(job, cur_time): 
                job.ddl_type = DDLState.BEST 
                job.cache_solution = None 


    def flush_jobs(self, prev_time, cur_time, status):
        if status == JobState.EVENT:
            self.flush_event_jobs(prev_time, cur_time)
        
        elif status == JobState.RUNNING:
            self.flush_running_jobs(prev_time, cur_time)
        
        elif status == JobState.PENDING:
            self.flush_pending_jobs(prev_time, cur_time)
        
        else:
            raise NotImplementedError

    def abstract_leaes_with_cache_solution(self, job_list, cur_time):
        # 1. init
        required_resource_list = list()
        required_lease_list = list()
        maximum_lease_list = list()
        reward_list = list() 
        in_block_list = list()

        no_cache_required_resource_list = list()
        no_cache_required_lease_list = list()
        no_cache_maximum_lease_list = list()
        no_cache_in_block_list = list()
        no_cache_reward_list = list() 

        # 2. job abstraction
        for job in job_list:
            required_resource = convert_to_bin(job.target_num_gpus)
            required_lease = int(math.ceil((job.application.max_progress - job.progress) / self.lease_term_interval))
            if self.is_best_effort(job):
                reward_list.append(BE_REWARD * 1.0 / math.ceil((job.application.max_progress - job.progress)))
                maximum_lease = required_lease = 1
                assert job.cache_solution is None, 'best effort is not allowed to have cached solutions'
            elif self.miss_ddl_job(job, cur_time):
                continue 
            else: 
                maximum_lease = compute_maximum_lease(job.ddl_time, self.lease_term_interval, self.cur_lease_index) 
                reward_list.append(SLO_REWARD)
            
            if job.cache_solution is not None: 
                assert len(job.cache_solution) == maximum_lease, 'cache solution should match length {} v.s. {}'.format(len(job.cache_solution), maximum_lease)
            
            required_resource_list.append(required_resource)
            required_lease_list.append(required_lease)
            maximum_lease_list.append(maximum_lease)
            in_block_list.append(job) 
        
        global_cache_solution =  [self.resource_by_lease[i] for i in range(max(maximum_lease_list))] if len(maximum_lease_list) > 0 else list()
        cnt = 0 
        for required_resource, required_lease, maximum_lease, reward, job  in zip(required_resource_list, required_lease_list, maximum_lease_list, reward_list, in_block_list):
            if job.cache_solution is None:
                cnt += 1
                no_cache_required_resource_list.append(required_resource)
                no_cache_required_lease_list.append(required_lease)
                no_cache_maximum_lease_list.append(maximum_lease)
                no_cache_in_block_list.append(job)
                no_cache_reward_list.append(reward) 
            else:
                maximum_lease = compute_maximum_lease(job.ddl_time, self.lease_term_interval, self.cur_lease_index)
                for idx, occupy in enumerate(job.cache_solution):
                    global_cache_solution[idx] -= int(occupy * job.target_num_gpus) 
                    if global_cache_solution[idx] < 0: 
                        import pdb; pdb.set_trace() 
                    assert global_cache_solution[idx] >= 0, 'cache solution should be no less than 0 at {}, job id {}'.format(global_cache_solution[idx], job.name)  
        return no_cache_required_resource_list, no_cache_required_lease_list, no_cache_maximum_lease_list, no_cache_in_block_list, no_cache_reward_list, global_cache_solution
     
    def job_comp_func(self, job, cur_time): 
        best_effort = self.is_best_effort(job) 
        if best_effort: 
            time_info = job.application.max_progress # * job.target_num_gpus
        else: 
            time_info = (job.ddl_time - job.application.max_progress - cur_time) # * job.target_num_gpus
        return (best_effort, time_info)
        
    def flush_lease_jobs(self, prev_time, cur_time): 
        # for job in self.running_jobs: 
        #     if self.is_best_effort(job): 
        #         if job.cache_solution is not None: 
        #             import pdb; pdb.set_trace() 

        best_effort_running_jobs = [job for job in self.running_jobs if self.is_best_effort(job)]
        runnable_jobs = sorted(self.pending_jobs + self.running_jobs, key=lambda job: self.job_comp_func(job, cur_time) ) 
        required_resource_list, required_lease_list, maximum_lease_list, \
            in_block_list, reward_list, global_cache_solution = self.abstract_leaes_with_cache_solution(runnable_jobs, cur_time)
        
        free_gpus = self.cluster_manager.check_free_gpus() + sum([job.target_num_gpus for job in best_effort_running_jobs])
        if len(required_lease_list) > 0 and free_gpus > 0: 
            solution_matrix = self.mip_solver.job_selection(required_resource_list, \
                        required_lease_list, maximum_lease_list, reward_list, global_cache_solution, self.resource_by_lease)
            
            should_run_jobs = list() 
            should_preempt_jobs = list()

            for idx, job in enumerate(in_block_list):
                assert job.status == JobState.PENDING or (self.is_best_effort(job) and job.status == JobState.RUNNING) 
                if solution_matrix[idx] is not None: 
                    if solution_matrix[idx][0] == 1: 
                        if job.status == JobState.PENDING: 
                            should_run_jobs.append(job) 
                        else: 
                            assert (job.status == JobState.RUNNING and self.is_best_effort(job)), 'this job should be best effort and running'

                    if solution_matrix[idx][0] == 0: 
                        if job.status == JobState.RUNNING: 
                            assert self.is_best_effort(job), 'we only preempt best effort job'
                            should_preempt_jobs.append(job) 

                    if not self.is_best_effort(job):
                        job.cache_solution = solution_matrix[idx]
                else:
                    if job.status == JobState.RUNNING: 
                        assert self.is_best_effort(job), 'we only preempt best effort job'
                        should_preempt_jobs.append(job) 


            # if len(in_block_list) > 20 and len(should_preempt_jobs) > 0: 
            #     for job in in_block_list: 
            #         print('name {}, best effort {}, ddl_time {}, progress {}, cur_time {}, placement {}, progress {}, delta {}'.format(job.name, \
            #             self.is_best_effort(job), job.ddl_time, job.application.max_progress, cur_time, job.placement, job.progress, self.job_comp_func(job, cur_time)))
                
            #     slo_gpu = sum([job.target_num_gpus for job in self.running_jobs if not self.is_best_effort(job)])
            #     import pdb; pdb.set_trace()


            # 3. release resource 
            for job in should_preempt_jobs: 
                self.execute_preempt(job, cur_time)
                self.running_jobs.remove(job)
                self.pending_jobs.append(job)

            # 4. allocate resource
            self.place_jobs(should_run_jobs, cur_time) 

            ncnt = 0
            # 5. update lease number
            for job in should_run_jobs:
                # if job.status == JobState.RUNNING: 
                if job.placement is not None: 
                    if not self.is_best_effort(job): 
                        job.expected_lease_count -= 1
                    else: 
                        job.cache_solution = None 
                    self.running_jobs.append(job)
                    self.pending_jobs.remove(job)
                else:
                    job.cache_solution = None
                    ncnt += 1
                    self.logger.info('job {}, gpu {} cannot find solution'.format(job.name, job.target_num_gpus))
            # self.logger.info('number of {} can not find a feasible solution in {}'.format(ncnt, len(should_run_jobs)))
        
        for job in self.running_jobs + self.pending_jobs:
            if job.cache_solution is not None:
                job.cache_solution = job.cache_solution[1:]
                if len(job.cache_solution) == 0:
                    job.cache_solution = None
        
        # global increase lease count
        self.cur_lease_index += 1

    def finish_all_jobs(self, ):
        return len(self.running_jobs) + len(self.event_jobs) + len(self.pending_jobs) == 0
    

    def run(self, ):
        cur_time = 0
        while not self.finish_all_jobs():
            prev_time = max(0, cur_time - self.scheduling_time_interval)
            self.cur_time = cur_time
            self.flush_jobs(prev_time, cur_time, status=JobState.RUNNING)
            self.flush_jobs(prev_time, cur_time, status=JobState.EVENT)
            self.flush_jobs(prev_time, cur_time, status=JobState.PENDING)

            if cur_time % self.lease_term_interval == 0:
                self.flush_lease_jobs(prev_time, cur_time)
                self.next_lease_time = self.lease_term_interval + self.scheduling_time_interval
                self.resource_by_lease = self.resource_by_lease[1:]
                self.resource_by_lease.append(self.resource_by_lease[-1])
            
            cur_time += self.scheduling_time_interval

            # record resource statistics
            resource_summary(self) 
        
        schedule_summary(self)
        

    def is_best_effort(self, job): 
        return job.ddl_type == DDLState.BEST
    
    def miss_ddl_job(self, job, cur_time):
        return (not self.is_best_effort(job) and cur_time + job.application.max_progress - job.progress > job.ddl_time)

    def flush_event_jobs(self, prev_time, cur_time):
        event_list = list()
        for event in self.event_jobs:
            if event.submission_time <= cur_time:
                assert event.submission_time >= prev_time
                event_list.append(event) 
        if len(event_list) == 0:
            self.submit_job_num_list.append(0)
            return 

        for event_job in event_list:
            self.execute_start_job(event_job, cur_time)
            self.event_jobs.remove(event_job)
            event_job.attained_lease_count = 0 
            event_job.expected_lease_count = int(math.ceil(event_job.application.max_progress / self.lease_term_interval))
            if self.is_best_effort(event_job):
                event_job.deadline_lease_count = event_job.expected_lease_count
            else: 
                event_job.deadline_lease_count = compute_maximum_lease(event_job.ddl_time, self.lease_term_interval, self.cur_lease_index)

            event_job.cache_solution = None 
        self.submit_job_num_list.append(len(event_list))
