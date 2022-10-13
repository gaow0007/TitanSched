import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from client.job.state import JobState
from .base import BaseScheduler
from .schedutils import resource_summary, schedule_summary


class YarnCSScheduler(BaseScheduler):
    def __init__(self, job_manager, cluster_manager, user_manager, placement, name, logger, **kwargs):
        super(YarnCSScheduler, self).__init__(job_manager, cluster_manager, user_manager, placement=placement, name=name, logger=logger)
        assert self.name == 'yarn-cs'
        self.pending_jobs = job_manager.pending_jobs
        self.running_jobs = job_manager.running_jobs
        self.event_jobs = job_manager.event_jobs
        self.completion_jobs = list()
        self.check_time_interval = kwargs.get('check_time_interval', 10)
        self.save_dir = kwargs.get('save_dir', 'result/')
        self.total_user_quota = dict()
        self.free_user_quota = dict()
        total_gpu_num = self.cluster_manager.check_total_gpus() 
        if self.user_manager is None: 
            self.total_user_quota = None 
            self.free_user_quota = None 
        else:     
            for user, user_share in self.user_manager.user_info:
                self.total_user_quota[user.name] = int(user_share * total_gpu_num)
                self.free_user_quota[user.name] = int(user_share * total_gpu_num)
        
    
    def finish_all_jobs(self, ):
        return len(self.event_jobs) + len(self.pending_jobs) + len(self.running_jobs) == 0
    

    def release_job_resource(self, job, status=JobState.END):
        if self.placement.name == 'gandiva':
            ret = self.cluster_manager.release_gandiva_job_resource(job, status)
        else:
            ret = self.cluster_manager.release_job_resource(job, status)
        job.release_resource()
        return ret
        

    def try_allocate_resoure(self, job):
        if self.placement.name in ['random', 'consolidate', 'gandiva', 'local_search']:
            ret = self.placement.place_jobs(job)
        else:
            raise NotImplementedError
        return ret
        

    # abstract
    def place_jobs(self, jobs):
        for job in jobs:
            self.placement(job)
        
    
    # abstract
    def check_resource(self, **kwargs):
        resource_name = kwargs.get('resource', 'gpu')
        assert isinstance(resource_name, str)
        if 'gpu' == resource_name:
            return self.CLUSTER.check_free_gpus()
        if 'cpu' == resource_name:
            return self.CLUSTER.check_free_cpus()
        
        raise NotImplementedError

    def move_to_pending(self, job): 
        job.last_check_time = job.submission_time 
        self.pending_jobs.append(job)
        job.status = JobState.PENDING

    
    def execute_start_job(self, start_job, cur_time):
        self.move_to_pending(start_job)
        self.logger.info('---- job[%s] is added  at time[%d]' % (str(start_job.name), cur_time))

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
                need_remove_jobs.append(job)
                self.completion_jobs.append(job) 
                if hasattr(job, 'user'): 
                    self.free_user_quota[job.user.name] += job.target_num_gpus

        for job in need_remove_jobs:
            self.running_jobs.remove(job)


    def flush_pending_jobs(self, prev_time, cur_time):
        need_run_jobs = list() 
        self.pending_jobs.sort(key=lambda e: e.submission_time)
        for job in self.pending_jobs: 
            job.step(seconds=cur_time - max(prev_time, job.submission_time))
            if hasattr(job, 'user'): 
                if self.free_user_quota[job.user.name] < job.target_num_gpus:
                    continue
            
            if self.try_allocate_resoure(job):  
                need_run_jobs.append(job)
                job.status = JobState.RUNNING
        

        for job in need_run_jobs: 
            self.pending_jobs.remove(job)
            self.running_jobs.append(job)
            self.logger.info('----job [%d] starts from pending' % job.name)
            if hasattr(job, 'user'): 
                self.free_user_quota[job.user.name] -= job.target_num_gpus


    def flush_jobs(self, prev_time, cur_time, status):
        if status == JobState.EVENT:
            self.flush_event_jobs(prev_time, cur_time)
        
        elif status == JobState.RUNNING:
            self.flush_running_jobs(prev_time, cur_time)
        
        elif status == JobState.PENDING:
            self.flush_pending_jobs(prev_time, cur_time) 

        else:
            raise NotImplementedError


    def run(self, ):
        cur_time = 0
        while not self.finish_all_jobs():
            prev_time = max(0, cur_time - self.check_time_interval)
            self.flush_jobs(prev_time, cur_time, status=JobState.RUNNING)
            self.flush_jobs(prev_time, cur_time, status=JobState.EVENT)
            self.flush_jobs(prev_time, cur_time, status=JobState.PENDING)
            cur_time += self.check_time_interval
            resource_summary(self)

        schedule_summary(self)