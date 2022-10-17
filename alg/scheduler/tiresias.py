import os, sys
import math
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from client.job.state import JobState
from .base import BaseScheduler
from .schedutils import resource_summary, schedule_summary

HIGH_QUEUE_PRIORITY=0


class TiresiasScheduler(BaseScheduler):
    def __init__(self, job_manager, cluster_manager, user_manager, placement, name, logger, **kwargs):
        super(TiresiasScheduler, self).__init__(job_manager, cluster_manager, user_manager, placement=placement, name=name, logger=logger)
        assert self.name == 'tiresias'
        self.pending_jobs = job_manager.pending_jobs
        self.running_jobs = job_manager.running_jobs
        self.runnable_jobs = job_manager.running_jobs
        self.event_jobs = job_manager.event_jobs
        self.completion_jobs = list()
        self.scheduling_time_interval = kwargs.get('scheduling_time_interval', 10)
        self.save_dir = kwargs.get('save_dir', 'result/')
        
        # dlas specified
        self.solve_starvation = kwargs.get('solve_starvation', 0) # supports avoiding starvation method
        self.set_queues(kwargs.get('num_queue'), kwargs.get('queue_limit'))
        self.hpo_search_time_interval = kwargs.get('hpo_search_time_interval', -1)
        self.search_algorithm = kwargs.get('search_algorithm', None)
        

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
        self.move_to_pending(start_job)
        start_job.queue_id = HIGH_QUEUE_PRIORITY
        self.queues[0].append(start_job) # new start job is appended into higest
        self.logger.info('---- job[%s] is added  at time[%d]' % (str(start_job.name), cur_time))

    
    def update_job_queue(self, job, **kwargs):
        if kwargs.get('name') == 'service':
            cur_qid = job.queue_id
            service = kwargs.get('service')
            if service >= self.queue_limit[cur_qid] and cur_qid < self.num_queue - 1:
                cur_qid += 1
            if cur_qid > 0 and self.queue_limit[cur_qid-1] > service:
                cur_qid -= 1
            
            if cur_qid != job.queue_id:
                self.queues[job.queue_id].remove(job)
                self.queues[cur_qid].append(job)
                job.queue_id = cur_qid
        
        elif kwargs.get('name') == 'starvation':
            solve_starvation = kwargs.get('solve_starvation')
            if job.pending_time >= int(job.running_time * solve_starvation): 
                if job not in self.queues[0]:
                    self.queues[0].append(job)
                    self.queues[job.queue_id].remove(job)
                    job.queue_id
        else:
            raise NotImplementedError


    def execute_preempt(self, job, cur_time):
        assert self.release_job_resource(job, status=JobState.PENDING) == True
        job.release_resource() 
        self.logger.info('running job {} is preempted at time {}'.format(job.name, cur_time))
    


    def reallocate_resource(self, total_num_gpu):
        should_run_jobs, should_preempt_jobs = list(), list()
        for idx, queue in enumerate(self.queues):
            assert all([x.queue_id == idx for x in queue])
            queue.sort(key=lambda x: x.submission_time)
            for job in queue:
                if total_num_gpu >= job.target_num_gpus:
                    if job.status == JobState.PENDING:
                        should_run_jobs.append(job)
                    if job.status in [JobState.PENDING, JobState.RUNNING]:
                        total_num_gpu -= job.target_num_gpus
                else:
                    if job.status == JobState.RUNNING:
                        should_preempt_jobs.append(job)

        return should_run_jobs, should_preempt_jobs
    
    
    def flush_running_jobs(self, prev_time, cur_time):
        
        need_remove_jobs = list()
        for job in self.running_jobs:
            job.step(cur_time - max(prev_time, job.submission_time))
            if job.completion_time is not None: 
                assert self.release_job_resource(job) == True
                job.status = JobState.END
                need_remove_jobs.append(job)
                self.completion_jobs.append(job)
                self.queues[job.queue_id].remove(job)
            else:
                self.update_job_queue(job, service=job.attained_service, name='service')
        
        for job in need_remove_jobs:
            self.running_jobs.remove(job)
    
    
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
            

    def flush_pending_jobs(self, prev_time, cur_time):
        for job in self.pending_jobs:
            job.step(seconds=cur_time - max(prev_time, job.submission_time))
            if self.solve_starvation > 0 and job.queue_id > 0:
                self.update_job_queue(job, solve_starvation=self.solve_starvation, name='starvation')


    def flush_runnable_jobs(self, prev_time, cur_time):
        self.runnable_jobs = self.running_jobs + self.pending_jobs
        should_run_jobs, should_preempt_jobs = self.reallocate_resource(self.cluster_manager.check_total_gpus())
        # release preempting jobs
        for job in should_preempt_jobs: 
            self.execute_preempt(job, cur_time)
            if job in self.running_jobs:
                self.running_jobs.remove(job)
            if job not in self.pending_jobs:
                self.pending_jobs.append(job)

        self.place_jobs(should_run_jobs, cur_time)


        for job in should_run_jobs:
            if job.placement is not None:
                if job in self.pending_jobs:
                    self.pending_jobs.remove(job)
                
                if job not in self.running_jobs:
                    self.running_jobs.append(job)
        
        for job in should_preempt_jobs:
            assert job.status == JobState.PENDING
        self.runnable_jobs = list()


    # abstract method
    def place_jobs(self, jobs, cur_time):
        jobs = sorted(jobs, key=lambda e: -e.target_num_gpus)
        for job in jobs:
            if not self.try_allocate_resoure(job): # TODO, make it allowed allocate across nodes
                continue
            
            self.logger.info('pending job {} is resumed at time {}'.format(job.name, cur_time))


    def set_queues(self, num_queue, queue_limit):
        """
        default: queues[0] is the highest priority queue
        """
        assert len(queue_limit) == num_queue, '# of queue should match with {}'.format(len(queue_limit))
        self.num_queue = num_queue
        self.queues = [list() for i in range(self.num_queue)]
        self.queue_limit = queue_limit


    def finish_all_jobs(self, ):
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
            if self.hpo_search_time_interval > 0 and cur_time % self.hpo_search_time_interval == 0: 
                self.search_hpo(cur_time) 

        schedule_summary(self)