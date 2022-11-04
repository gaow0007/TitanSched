import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..') 
from client.job.state import JobState
from .base import BaseScheduler
from .schedutils import resource_summary, schedule_summary


def get_job_weighted_share(job, jobs):
    if len(jobs) == 0:
        return float('inf')
    if 'weight' not in job:
        return 1. / len(jobs)
    return 1. * job['weight'] / sum(job['weight'] for job in jobs)


class ThemisScheduler(BaseScheduler):
    def __init__(self, job_manager, cluster_manager, user_manager, placement, name, logger, **kwargs):
        super(ThemisScheduler, self).__init__(job_manager, cluster_manager, user_manager, placement=placement, name=name, logger=logger)
        assert self.name == 'themis'
        self.pending_jobs = job_manager.pending_jobs
        self.running_jobs = job_manager.running_jobs
        self.event_jobs = job_manager.event_jobs
        self.completion_jobs = list()
        self.scheduling_time_interval = kwargs.get('scheduling_time_interval', 10)
        self.lease_term_interval = kwargs.get('lease_term_interval', 300)
        self.save_dir = kwargs.get('save_dir', 'result/')
    

    def finish_all_jobs(self, ):
        # print('event_jobs {}, pending_jobs {}, running_jobs {}'.format(len(self.event_jobs), len(self.pending_jobs), len(self.running_jobs)))
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
            return self.CLUSTER.check_free_gpus()
        if 'cpu' == resource_name:
            return self.CLUSTER.check_free_cpus()
        
        raise NotImplementedError
    
    def execute_preempt(self, job, cur_time):
        assert self.release_job_resource(job, status=JobState.PENDING) == True
        job.release_resource() 
        self.logger.info('running job {} is preempted at time {}'.format(job.name, cur_time))

    def execute_start_job(self, start_job, cur_time):
        self.pending_jobs.append(start_job)
        start_job.status = JobState.PENDING
        self.logger.info('---- job[{}] is added  at time[{}]'.format(start_job.name, cur_time))
    
    def check_lease_expire(self, job, cur_time): 
        return  cur_time - job.last_running_time >= self.lease_term_interval

    def get_finish_time_fairness(self, job):
        if job.deserved_service == 0: 
            return 1 
        t_remaining = job.predict_remaining_time(job.target_num_gpus)
        t_isolated = job.running_time * (job.attained_service / job.deserved_service) + (job.staying_time - job.running_time)
        return job.staying_time / (t_isolated + t_remaining)


    def update_job_fairness(self, prev_time, cur_time): 
        total_job_num = len(self.pending_jobs) + len(self.running_jobs)
        # total_gpu_request = sum([job.target_num_gpus for job in (self.pending_jobs + self.running_jobs)])
        total_gpu_request = len(self.pending_jobs) + len(self.running_jobs)
        weighted_share = self.cluster_manager.check_total_gpus() / (total_job_num + 1e-3)
        for job in self.pending_jobs + self.running_jobs: 
            interval = cur_time - max(prev_time, job.submission_time)
            if not hasattr(job, 'deserved_service'): 
                job.deserved_service = 0 

            job.deserved_service += min(weighted_share, job.target_num_gpus) * interval 


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
        for job in self.running_jobs:
            job.step(cur_time - max(prev_time, job.submission_time)) 
            if job.completion_time is not None: 
                assert self.release_job_resource(job) == True 
                job.status = JobState.END
                job.finish_time_fairness = self.get_finish_time_fairness(job)
                self.completion_jobs.append(job)
                need_remove_jobs.append(job)
                self.completion_jobs.append(job)
    
        for job in need_remove_jobs:
            self.running_jobs.remove(job)


    def flush_pending_jobs(self, prev_time, cur_time):
        for job in self.pending_jobs:
            job.step(seconds=cur_time - max(prev_time, job.submission_time))
    

    def flush_runnable_jobs(self, prev_time, cur_time): 

        # 1. sorted by shortest remaining time 
        for job in self.pending_jobs: job.finish_time_fairness = self.get_finish_time_fairness(job)
        self.runnable_jobs = [job for job in self.pending_jobs] # need enumerate, watch out shallow copy and deep copy 
        total_resource_count = self.cluster_manager.check_free_gpus()
        for job in self.running_jobs: 
            if hasattr(job, 'preemptible') and job.preemptible and self.check_lease_expire(job, cur_time): 
                job.finish_time_fairness = self.get_finish_time_fairness(job)
                self.runnable_jobs.append(job) 
                total_resource_count += job.target_num_gpus 
        
        self.runnable_jobs.sort(key=lambda e: e.finish_time_fairness ** (-1))
        
        # 2. select which jobs to run or preempty
        
        should_run_jobs, should_preempt_jobs = list(), list()

        for job in self.runnable_jobs:
            # if job.finish_time_fairness < 0.9: 
            #     continue 
            if job.target_num_gpus <= total_resource_count:
                total_resource_count -= job.target_num_gpus
                if job.status == JobState.PENDING:
                    should_run_jobs.append(job)
                elif job.status == JobState.RUNNING: 
                    job.last_running_time = job.last_running_time + self.lease_term_interval
                    pass
                else: 
                    raise NotImplementedError
            elif job.status == JobState.RUNNING:
                should_preempt_jobs.append(job)

        # 3. release resource
        for job in should_preempt_jobs: 
            self.execute_preempt(job, cur_time)
            if job in self.running_jobs:
                self.running_jobs.remove(job)
            if job not in self.pending_jobs:
                self.pending_jobs.append(job)
        
        # 4. allocate resource 
        self.place_jobs(should_run_jobs, cur_time)
        for job in should_run_jobs:
            if job.placement is not None:
                if job in self.pending_jobs:
                    self.pending_jobs.remove(job)
                if job not in self.running_jobs:
                    self.running_jobs.append(job) 
                job.last_running_time = cur_time 

        self.runnable_jobs = list()



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
            self.update_job_fairness(prev_time, cur_time)
            self.flush_jobs(prev_time, cur_time, status=JobState.RUNNABLE)
            cur_time += self.scheduling_time_interval
            # record resource statistics
            resource_summary(self) 
        
        schedule_summary(self)
        
