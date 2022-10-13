
import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from .base import BaseScheduler
from .disc_priority import DiscPrioirtyScheduler
from utils.util import search_dict_list
import numpy as np
import math

CHECKPOINT_OVER_HEAD = 0.1
RESUME_OVER_HEAD = 0.4

MAX_EXPECT_VALUE = 10


class OptimusScheduler(BaseScheduler):
    def __init__(self, JOBS, CLUSTER, USERS, placement, name, logger, **kwargs):
        super(OptimusScheduler, self).__init__(JOBS=JOBS, CLUSTER=CLUSTER, USERS=USERS, placement=placement, name=name, logger=logger)
        assert self.name == 'optimus'
        self.pending_jobs = JOBS.pending_jobs
        self.running_jobs = JOBS.running_jobs
        self.runnable_jobs = JOBS.runnable_jobs # pending + running
        self.event_jobs = JOBS.job_events # collect on-going ending jobs
        self.end_jobs = list()
        self.end_events = list() 
        self.check_time_interval = kwargs.get('check_time_interval', 10)
        self.save_dir = kwargs.get('save_dir', 'result/')
        self.cur_time = 0



    def move_to_pending(self, job):
        ''' job gets into the system: pending or running, and finally END'''
        job['last_check_time'] = job['submit_time']
        self.pending_jobs.append(job)
        job['status'] = 'PENDING'
        job['start_time'] = sys.maxsize


    def execute_start_job(self, start_job, cur_time):
        self.move_to_pending(start_job)
        self.logger.info('---- job[%d] is added  at time[%d]' % (start_job['job_id'], cur_time))


    # abstract method
    def check_resource(self, **kwargs):
        resource_name = kwargs.get('resource', 'gpu')
        assert isinstance(resource_name, str)
        if 'gpu' == resource_name:
            return self.CLUSTER.check_free_gpus()
        if 'cpu' == resource_name:
            return self.CLUSTER.check_free_cpus()
        
        raise NotImplementedError
    

    def batch_place_jobs(self, jobs, cur_time):
        assert self.placement.name == 'local_search'
        remaining_gpu_num = self.CLUSTER.check_free_gpus()
        filter_jobs = list()
        for job in jobs:
            if remaining_gpu_num >= job['num_gpu']:
                remaining_gpu_num -= job['num_gpu']
                filter_jobs.append(job)
        if len(filter_jobs) == 0:
            return 
        import time
        start = time.time()
        assert self.placement.place_jobs(filter_jobs) == True
        self.logger.info('elapsed time {}'.format(time.time() - start))

        for job in filter_jobs:
            job['resume'] += 1
            job['status'] = 'RUNNING'
            if job['start_time'] == sys.maxsize:
                job['start_time'] = cur_time
            self.logger.info('pending job {} is resumed at time {}'.format(job['job_id'], cur_time))


    def try_allocate_resoure(self, job):
        if self.placement.name in ['random', 'consolidate', 'gandiva', 'local_search', 'consolidate_random']:
            ret = self.placement.place_jobs(job)
        else:
            raise NotImplementedError
        return ret

    # abstract
    def place_jobs(self, jobs, cur_time):
        if self.placement.name == 'local_search':
            self.batch_place_jobs(jobs, cur_time)
            return 
        
        for job in jobs:
            if not self.try_allocate_resoure(job): # TODO, make it allowed allocate across nodes
                continue

            job['resume'] += 1
            job['status'] = 'RUNNING'
            if job['start_time'] == sys.maxsize:
                job['start_time'] = cur_time
            
            self.logger.info('pending job {} is resumed at time {}'.format(job['job_id'], cur_time))


    def release_job_resource(self, job, status='END'):
        if self.placement.name == 'gandiva':
            ret = self.CLUSTER.release_gandiva_job_resource(job, status)
        else:
            ret = self.CLUSTER.release_job_resource(job, status)
            
        return ret


    def execute_preempt(self, job, cur_time):
        job['preempt'] += 1
        job['status'] = 'PENDING'
        job['last_pending_time'] = 0
        assert self.release_job_resource(job, status='PENDING') == True
        self.logger.info('running job {} is preempted at time {}'.format(job['job_id'], cur_time))


    def flush_runnable_jobs(self, prev_time, cur_time):
        should_run_jobs, should_preempt_jobs = list(), list()
        runnable_jobs = self.pending_jobs + self.running_jobs 

        if len(runnable_jobs) == 0:
            return should_run_jobs, should_preempt_jobs
        
        total_num_gpu = self.CLUSTER.check_total_gpus()
        runnable_jobs.sort(key=lambda x: x.__getitem__('submit_time'))
        min_replicas = [1 for job in runnable_jobs]
        num_replicas = [0 for job in runnable_jobs]
        gains = [0 for job in runnable_jobs]
        for job_idx, job in enumerate(runnable_jobs): 
            if min_replicas[job_idx] > total_num_gpu: 
                continue 
            num_replicas[job_idx] = min_replicas[job_idx]
            total_num_gpu -= min_replicas[job_idx]
            remaining = job['duration'] - job['progress']
            gains[job_idx] = remaining / self.predict_speedup(job, num_replicas[job_idx]) - remaining / self.predict_speedup(job, num_replicas[job_idx] + 1)

        
        while total_num_gpu > 0 and max(gains) > 0: 
            job_idx = np.argmax(gains)
            num_replicas[job_idx] += 1
            total_num_gpu -= 1
            if num_replicas[job_idx] + 1 > job['max_gpu_num']: 
                gains[job_idx] = 0
                continue 
            remaining = job['duration'] - job['progress']
            gains[job_idx] = remaining / self.predict_speedup(job, num_replicas[job_idx]) - remaining / self.predict_speedup(job, num_replicas[job_idx] + 1)
        
        
        for job_idx, job in enumerate(runnable_jobs): 
            if num_replicas[job_idx] != job['allocated_gpu_num']: 
                # self.logger.info('job {} is preempted at time {}'.format(job['job_id'], cur_time))
                if job['status'] == 'RUNNING':
                    self.execute_preempt(job, cur_time)

                if job in self.running_jobs:
                    self.running_jobs.remove(job)
                if job not in self.pending_jobs:
                    self.pending_jobs.append(job)
            
            if num_replicas[job_idx] > 0: 
                if num_replicas[job_idx] != job['allocated_gpu_num']:
                    should_run_jobs.append(job)
                    job['required_gpu_num'] = num_replicas[job_idx]
                    job['num_gpu'] = num_replicas[job_idx]
            else: 
                job['allocated_gpu_num'] = 0 

        self.place_jobs(should_run_jobs, cur_time)

        for job in should_run_jobs: 
            if job['status'] == 'RUNNING': 
                job_idx = runnable_jobs.index(job)
                job['allocated_gpu_num'] = num_replicas[job_idx]
            else: 
                job['allocated_gpu_num'] = 0

        for job in runnable_jobs: 
            if job['status'] == 'RUNNING': 
                if job in self.pending_jobs: 
                    self.pending_jobs.remove(job)
                if job not in self.running_jobs: 
                    self.running_jobs.append(job)
            
            if job['status'] == 'PENDING': 
                if job in self.running_jobs: 
                    self.running_jobs.remove(job)
                if job not in self.pending_jobs: 
                    self.pending_jobs.append(job)
                

    def flush_running_jobs(self, prev_time, cur_time):
        need_remove_jobs = list()
        for job in self.running_jobs:
            time_diff = int(cur_time - job['last_check_time'])
            speedup = self.predict_speedup(job, job["allocated_gpu_num"]) 
            progress = time_diff * speedup
            if progress + job['progress'] >= job['duration']:
                assert self.release_job_resource(job) == True
                job['end_time'] = int(job['last_check_time'] + (job['duration'] - job['progress']) / speedup + job['preempt'] * (CHECKPOINT_OVER_HEAD + RESUME_OVER_HEAD) * (1 + math.log2(job['true_gpu_num'])))
                self.update_job_time(job, job['end_time'], name='RUNNING')
                self.logger.info('complete job {} at time {}'.format(job['job_id'], job['end_time']))
                job['status'] = 'END'
                need_remove_jobs.append(job)
                self.end_jobs.append(job) 
            else: 
                self.update_job_time(job, cur_time, name='RUNNING') 
                job['progress'] += time_diff 

        for job in need_remove_jobs:
            self.running_jobs.remove(job)
    
    

    def predict_speedup(self, job, num_gpu):
        if num_gpu >= job['true_gpu_num']: 
            return (1.0 * num_gpu  / job['true_gpu_num']) ** job["power"]
        else: 
            return (1.0 * num_gpu  / job['true_gpu_num'])


    def update_job_time(self, job, cur_time, name):
        if name == 'RUNNING':
            delta_time = int(cur_time - job['last_check_time'])
            job['total_executed_time'] = int(job['total_executed_time'] + delta_time)
            job['executed_time'] = int(job['executed_time'] + delta_time)
            job['last_check_time'] = cur_time
        elif name == 'PENDING':
            delta_time = int(cur_time - job['last_check_time'])
            job['pending_time'] += delta_time
            job['last_check_time'] = cur_time
        else:
            raise NotImplementedError

    def flush_pending_jobs(self, prev_time, cur_time):
        need_remove_jobs = list() 
        for job in self.pending_jobs:
            self.update_job_time(job, cur_time, name='PENDING')

        for job in need_remove_jobs:
            self.pending_jobs.remove(job)
    
    def flush_event_jobs(self, prev_time, cur_time):
        event_list = list()
        for event in self.event_jobs:
            if event['time'] <= cur_time:
                assert event['time'] >= prev_time
                event_list.append(event)
        
        submit_job_num = 0
        for event in event_list:
            for start_job in event['start_jobs']:
                self.execute_start_job(start_job, cur_time)
                start_job['last_check_time'] = cur_time
                start_job['pending_time'] = cur_time - start_job['submit_time']
                # start_job['estimate_duration'] = max(start_job['duration'] * (0.9 + 0.2 * np.random.rand()) + self.mean_duration * (np.random.rand() - 0.5), 1)
                start_job['true_expect_time'] = start_job['expect_time'] + start_job['submit_time']
                start_job['true_gpu_num'] = start_job.required_gpu_num
                start_job['required_gpu_num'] = 0 
                start_job['num_gpu'] = 0
                start_job['allocated_gpu_num'] = 0
                start_job["power"] = 0.1 +  np.random.rand() * 0.3
                start_job['progress'] = 0 # start_job['duration']
                start_job['max_gpu_num'] = 128 
                submit_job_num += 1
            self.event_jobs.remove(event)

        self.submit_job_num_list.append(submit_job_num)


    def best_effort_job(self, job, cur_time):
        return ('best_effort' in job and job['best_effort'] == 1) 
    
    
    def flush_jobs(self, prev_time, cur_time, status):
        if status == 'EVENT':
            self.flush_event_jobs(prev_time, cur_time)
        
        elif status == 'RUNNING':
            self.flush_running_jobs(prev_time, cur_time)
        
        elif status == 'PENDING':
            self.flush_pending_jobs(prev_time, cur_time)
        
        elif status == 'RUNNABLE':
            self.flush_runnable_jobs(prev_time, cur_time)
        
        else:
            raise NotImplementedError

    def finish_all_jobs(self, ):
        return len(self.running_jobs) + len(self.event_jobs) + len(self.pending_jobs) == 0


    def run(self, ):
        cur_time = 0
        while not self.finish_all_jobs():
            prev_time = max(0, cur_time - self.check_time_interval)
            self.flush_jobs(prev_time, cur_time, status='RUNNING')
            self.flush_jobs(prev_time, cur_time, status='EVENT')
            self.flush_jobs(prev_time, cur_time, status='PENDING')
            self.flush_jobs(prev_time, cur_time, status='RUNNABLE')
            cur_time += self.check_time_interval
            self.cur_time = cur_time
            self.resource_summary()
            
        self.schedule_summary()


    def schedule_summary(self, ):
        assert all([job['status'] == 'END' for job in self.JOBS.job_list])
        attribute_list = ['job_id', 'pending_time', 'total_executed_time', 'submit_time', 'end_time', 'preempt', 'resume', 'num_gpu', 'expect_time', 'miss_ddl', 'expect_time_list', 'expect_value_list', 'best_effort']
        
        for job in self.JOBS.job_list:
            job['miss_ddl'] = 0
            if 'best_effort' not in job or job['best_effort'] == 0:
                job['miss_ddl'] = 1
                for vid, (expect_time, expect_value) in enumerate(zip(job['expect_time_list'], job['expect_value_list'])):
                    if expect_time + job['submit_time'] >= job['end_time']:
                        job['miss_ddl'] = 1 - expect_value * 1.0 / MAX_EXPECT_VALUE
                        break
            job['expect_time_list'] =  '-'.join([str(item) for item in job['expect_time_list']])
            job['expect_value_list'] = '-'.join([str(item) for item in job['expect_value_list']])
            
        with open(os.path.join(self.save_dir, self.name + '.csv'), 'w') as f:
            print(",".join(attribute_list), file=f)
            for job in self.JOBS.job_list:
                for attribute in attribute_list:
                    print("{}".format(job[attribute]), file=f) if attribute == attribute_list[-1] else print("{}".format(job[attribute]), end=',', file=f)
        
        attribute_list = ['full_resource', 'free_resource', 'pending_num', 'running_num', 'submit_num']
        with open(os.path.join(self.save_dir, self.name + '_resource.csv'), 'w') as f:
            print(",".join(attribute_list), file=f)
            for full_resource, free_resource, pending_num, running_num, submit_num in zip(self.full_resource_list, self.free_resource_list, self.pending_job_num_list, self.running_job_num_list, self.submit_job_num_list):
                print('{},{},{},{},{}'.format(full_resource, free_resource, pending_num, running_num, submit_num), file=f)


    def resource_summary(self, ):
        self.full_resource_list.append(self.CLUSTER.check_total_gpus())
        self.free_resource_list.append(self.CLUSTER.check_free_gpus())
        self.pending_job_num_list.append(len(self.pending_jobs))
        self.running_job_num_list.append(len(self.running_jobs))