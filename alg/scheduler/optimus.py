import os, sys
import numpy as np 
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..') 
from client.job.state import JobState
from .base import BaseScheduler
from .schedutils import resource_summary, schedule_summary


class OptimusScheduler(BaseScheduler):
    def __init__(self, job_manager, cluster_manager, user_manager, placement, name, logger, **kwargs):
        super(OptimusScheduler, self).__init__(job_manager, cluster_manager, user_manager, placement=placement, name=name, logger=logger)
        assert self.name == 'optimus'
        self.pending_jobs = job_manager.pending_jobs
        self.running_jobs = job_manager.running_jobs
        self.event_jobs = job_manager.event_jobs
        self.completion_jobs = list()
        self.scheduling_time_interval = kwargs.get('scheduling_time_interval', 10)
        self.save_dir = kwargs.get('save_dir', 'result/') 
        

    # abstract method
    def check_resource(self, **kwargs):
        resource_name = kwargs.get('resource', 'gpu')
        assert isinstance(resource_name, str)
        if 'gpu' == resource_name:
            return self.CLUSTER.check_free_gpus()
        if 'cpu' == resource_name:
            return self.CLUSTER.check_free_cpus()
        
        raise NotImplementedError
    

    def try_allocate_resoure(self, job):
        if self.placement.name in ['random', 'consolidate', 'gandiva', 'local_search', 'consolidate_random']:
            ret = self.placement.place_jobs(job)
        else:
            raise NotImplementedError
        return ret


    def release_job_resource(self, job, status=JobState.END):
        if self.placement.name == 'gandiva':
            ret = self.cluster_manager.release_gandiva_job_resource(job, status)
        else:
            ret = self.cluster_manager.release_job_resource(job, status)
        job.release_resource()
        return ret
        

    def move_to_pending(self, job):
        ''' job gets into the system: pending or running, and finally END'''
        job.last_check_time = job.submission_time 
        self.pending_jobs.append(job)
        job.status = JobState.PENDING

    
    def execute_start_job(self, start_job, cur_time):
        self.pending_jobs.append(start_job)
        start_job.target_num_gpus = 0 
        start_job.status = JobState.PENDING
        self.logger.info('---- job[{}] is added  at time[{}]'.format(start_job.name, cur_time))


    def execute_preempt(self, job, cur_time):
        assert self.release_job_resource(job, status=JobState.PENDING) == True
        job.release_resource() 
        self.logger.info('running job {} is preempted at time {}'.format(job.name, cur_time))
    

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
        
        for job in need_remove_jobs:
            self.running_jobs.remove(job)


    def flush_pending_jobs(self, prev_time, cur_time):
        for job in self.pending_jobs:
            job.step(seconds=cur_time - max(prev_time, job.submission_time))

    def flush_runnable_jobs(self, prev_time, cur_time):
        runnable_jobs = self.running_jobs + self.pending_jobs
        if len(runnable_jobs) == 0:
            return 
        runnable_jobs.sort(key=lambda job:job.predict_remaining_time(1))

        # init
        should_run_jobs = list() 
        total_num_gpu = self.cluster_manager.check_total_gpus() 
        runnable_jobs.sort(key=lambda job: job.submission_time)
        # min_replicas = [job.application.min_num_gpus for job in runnable_jobs]
        num_replicas = [0 for job in runnable_jobs]
        gains = [0 for job in runnable_jobs]

        if False: 
            # sort jobs and assgin gpus
            for job_idx, job in enumerate(runnable_jobs): 
                # if min_replicas[job_idx] > total_num_gpu: 
                #     continue 
                num_replicas[job_idx] = 0
                gains[job_idx] = job.predict_loss(num_replicas[job_idx]) - job.predict_loss(num_replicas[job_idx] + 1)
            
            while total_num_gpu > 0 and max(gains) > 0: 
                job_idx = np.argmax(gains)
                num_replicas[job_idx] += 1
                total_num_gpu -= 1
                if num_replicas[job_idx] + 1 > job.max_num_gpus: 
                    gains[job_idx] = 0
                    continue 
                
                gains[job_idx] = job.predict_loss(num_replicas[job_idx]) - job.predict_loss(num_replicas[job_idx] + 1)
        else: 
            min_replicas = [job.min_num_gpus for job in runnable_jobs]
            # sort jobs and assgin gpus
            for job_idx, job in enumerate(runnable_jobs): 
                if min_replicas[job_idx] > total_num_gpu: 
                    num_replicas[job_idx] = 0
                    gains[job_idx] = 0 
                    continue 
                num_replicas[job_idx] = min_replicas[job_idx]
                total_num_gpu -= min_replicas[job_idx]
                gains[job_idx] = job.predict_remaining_time(num_replicas[job_idx]) - job.predict_remaining_time(num_replicas[job_idx] + 1)
                # import pdb; pdb.set_trace() 


            while total_num_gpu > 0 and max(gains) > 0: 
                job_idx = np.argmax(gains)
                num_replicas[job_idx] += 1
                total_num_gpu -= 1
                if num_replicas[job_idx] + 1 > job.max_num_gpus: 
                    gains[job_idx] = 0
                    continue 
                
                gains[job_idx] = job.predict_remaining_time(num_replicas[job_idx]) - job.predict_remaining_time(num_replicas[job_idx] + 1)
        
        for job_idx, job in enumerate(runnable_jobs): 
            if num_replicas[job_idx] != job.target_num_gpus: 
                # self.logger.info('job {} is preempted at time {}'.format(job['job_id'], cur_time))
                if job.status == JobState.RUNNING:
                    self.execute_preempt(job, cur_time)

                if job in self.running_jobs:
                    self.running_jobs.remove(job)
                if job not in self.pending_jobs:
                    self.pending_jobs.append(job)
            
            if num_replicas[job_idx] > 0: 
                if num_replicas[job_idx] != job.target_num_gpus:
                    should_run_jobs.append(job)
                    job.target_num_gpus = num_replicas[job_idx]
            else: 
                job.target_num_gpus = 0

        self.place_jobs(should_run_jobs, cur_time)

        for job in should_run_jobs: 
            if job.placement is not None: 
                job.status = JobState.RUNNING
            else: 
                job.status = JobState.PENDING

            if job.status == JobState.RUNNING:
                job_idx = runnable_jobs.index(job)
                job.target_num_gpus = num_replicas[job_idx]
            else: 
                job.target_num_gpus = 0

        for job in runnable_jobs: 
            if job.status == JobState.RUNNING:
                if job in self.pending_jobs: 
                    self.pending_jobs.remove(job)
                if job not in self.running_jobs: 
                    self.running_jobs.append(job)

            if job.status == JobState.PENDING:
                if job in self.running_jobs: 
                    self.running_jobs.remove(job)
                if job not in self.pending_jobs: 
                    self.pending_jobs.append(job)


    # abstract method
    def place_jobs(self, jobs, cur_time):
        jobs = sorted(jobs, key=lambda e: -e.target_num_gpus)
        for job in jobs:
            if not self.try_allocate_resoure(job): # TODO, make it allowed allocate across nodes
                continue
            
            self.logger.info('pending job {} is resumed at time {}, and allocated {} gpu'.format(job.name, cur_time, job.target_num_gpus))



    def finish_all_jobs(self, ):
        # print('running {}, event {}, pending {}'.format(len(self.running_jobs), len(self.event_jobs), len(self.pending_jobs)))
        return len(self.running_jobs) + len(self.event_jobs) + len(self.pending_jobs) == 0


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
            resource_summary(self)

        schedule_summary(self)