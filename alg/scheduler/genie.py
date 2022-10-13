
import os, sys
from tracemalloc import start
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from .base import BaseScheduler
from utils.util import search_dict_list
import numpy as np
import math

CHECKPOINT_OVER_HEAD = 0.1
RESUME_OVER_HEAD = 0.4

MAX_EXPECT_VALUE = 10


class GenieScheduler(BaseScheduler):
    def __init__(self, JOBS, CLUSTER, USERS, placement, name, logger, **kwargs):
        super(GenieScheduler, self).__init__(JOBS=JOBS, CLUSTER=CLUSTER, USERS=USERS, placement=placement, name=name, logger=logger)
        assert self.name == 'genie'
        self.pending_jobs = JOBS.pending_jobs
        self.running_jobs = JOBS.running_jobs
        self.runnable_jobs = JOBS.runnable_jobs # pending + running
        self.event_jobs = JOBS.job_events # collect on-going ending jobs
        self.end_jobs = list()
        self.end_events = list() 
        self.check_time_interval = kwargs.get('check_time_interval', 10)
        self.save_dir = kwargs.get('save_dir', 'result/')
        self.mean_duration = 20
        self.solve_starvation = kwargs.get('solve_starvation', 0) # supports avoiding starvation method
        self.set_queues(kwargs.get('num_queue'), kwargs.get('queue_limit'))
        self.cur_time = 0


    def calculate_time_aware_queue_id(self, job, time_tag):
        assert job['usr_mode'] == 'deadline'
        expected_remaining_time = job['ddl_time'] - time_tag 
        true_remaining_time = job['duration'] - job['total_executed_time']
        service = 1. * true_remaining_time / (expected_remaining_time + 1e-3)
        queue_id = 0
        while service > self.queue_limit[queue_id] and queue_id < len(self.queue_limit) - 1:
            queue_id += 1
        
        return queue_id

    def execute_start_job(self, start_job, cur_time):
        self.move_to_pending(start_job)
        queue_id = self.calculate_time_aware_queue_id(start_job, cur_time)
        start_job['queue_id'] = queue_id 
        self.queues[queue_id].append(start_job)
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

    def miss_ddl_job(self, job, cur_time):
        # return False
        return ('ddl_type' in job and job['ddl_type'] != 'best_effort')  and (job.uesr_specified_deadline_time + job.submit_time < cur_time + job['fake_duration'] - job['total_executed_time'])


    def flush_running_jobs(self, prev_time, cur_time):
        need_remove_jobs = list()
        for job in self.running_jobs:
            time_diff = int(cur_time - job['last_check_time'])
            if self.miss_ddl_job(job, cur_time):
                assert self.release_job_resource(job) == True
                job['end_time'] =  job.uesr_specified_deadline_time * 2 + job.submit_time 
                job['status'] = 'END'
                self.end_jobs.append(job) 
                need_remove_jobs.append(job)
            elif time_diff + job['total_executed_time'] >= job['fake_duration']:
                assert self.release_job_resource(job) == True
                job['end_time'] = job['last_check_time'] + job['fake_duration'] - job['total_executed_time'] + job['preempt'] * (CHECKPOINT_OVER_HEAD + RESUME_OVER_HEAD) * (1 + math.log2(job.required_gpu_num))
                self.update_job_time(job, job['end_time'], name='RUNNING')
                self.logger.info('complete job {} at time {}'.format(job['job_id'], job['end_time']))
                job['status'] = 'END'
                need_remove_jobs.append(job)
                self.queues[job['queue_id']].remove(job)
            else:
                self.update_job_time(job, cur_time, name='RUNNING')

        for job in need_remove_jobs:
            self.running_jobs.remove(job)
    

    def update_job_emergence(self, cur_time):
        self.runnable_jobs = self.pending_jobs + self.running_jobs
        for job in self.runnable_jobs:
            assert 'usr_mode' in job  
            job['emergence'] = job['ddl_time'] - job['submit_time'] - cur_time
            if 'ddl_type' in job and job['ddl_type'] == 'best_effort':
                job['emergence'] = 0
            # check demotion or promotion in queue
            self.update_job_queue(job, service=job['emergence'], name='service')
    
    
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
                start_job['estimate_duration'] = max(start_job['duration'] * (0.9 + 0.2 * np.random.rand()) + self.mean_duration * (np.random.rand() - 0.5), 1)
                if self.best_effort_job(start_job, cur_time=cur_time): 
                    emergence =  start_job['submit_time']
                else:
                    emergence = start_job['ddl_time'] - start_job['estimate_duration'] - cur_time
                    
                start_job['true_gpu_num'] = start_job.required_gpu_num
                start_job['emergence'] = emergence
                start_job['fake_emergence'] = start_job['emergence']
                start_job['fake_gpu_num'] = start_job.required_gpu_num
                start_job['fake_duration'] = start_job['duration']
                start_job['fake_estimate_duration'] = start_job['estimate_duration']
                submit_job_num += 1
            self.event_jobs.remove(event)

        self.submit_job_num_list.append(submit_job_num)


    def best_effort_job(self, job, cur_time):
        return 'ddl_type' in job and job['ddl_type'] == 'best_effort'
    
    def calculate_ces(self, job, total_num_gpu):
        job['emergence'] = job['ddl_time'] - job['estimate_duration'] - self.cur_time
        job['fake_emergence'] = job['emergence']
        job['fake_gpu_num'] = job.required_gpu_num
        job['fake_duration'] = job['duration']
        job['fake_estimate_duration'] = job['estimate_duration']


        benefit = sys.maxsize
        for idx in range(10):
            gpu_num = 2 ** idx
            if gpu_num <= job.required_gpu_num * 4 and gpu_num >= job.required_gpu_num // 4 and gpu_num <= total_num_gpu and gpu_num != job.required_gpu_num:
                ratio = job.required_gpu_num * 1.0 / gpu_num
                fake_estimate_duration = ratio * job['estimate_duration'] + (np.random.rand() - 0.5) * self.mean_duration
                fake_duration = fake_estimate_duration #  ratio * job['duration'] + (np.random.rand() - 0.5) * self.mean_duration
                emergence = (job['ddl_time'] - fake_estimate_duration - self.cur_time) * gpu_num
                cal_benefit = fake_duration * gpu_num
                if self.best_effort_job(job, self.cur_time):
                    if cal_benefit < benefit:
                        benefit = cal_benefit
                        job['fake_gpu_num'] = gpu_num
                        job['fake_duration'] = fake_duration
                        job['fake_estimate_duration'] = fake_estimate_duration
                else:
                    if emergence >= 0 and cal_benefit < benefit:
                        benefit = cal_benefit
                        job['fake_emergence'] = emergence
                        job['fake_gpu_num'] = gpu_num
                        job['fake_duration'] = fake_duration
                        job['fake_estimate_duration'] = fake_estimate_duration
                

    def reallocate_resource(self, total_num_gpu):
        should_run_jobs, should_preempt_jobs = list(), list()
        total_num_gpu = self.CLUSTER.check_free_gpus()
        if total_num_gpu == 0:
            return should_run_jobs, should_preempt_jobs
        for job in self.pending_jobs:
            if 'ddl_type' in job and job['ddl_type'] == 'best_effort':
                job['emergence'] = job['submit_time']
            else:
                self.calculate_ces(job, total_num_gpu)

        self.pending_jobs.sort(key=lambda x: (x.__getitem__('ddl_type') == 'best_effort', x.__getitem__('fake_emergence'))) # , x.__getitem__('submit_time')))
        for job in self.pending_jobs:
            if not self.best_effort_job(job, self.cur_time) and job['fake_emergence'] >= 0:
                if job['fake_emergence'] != job['emergence'] and total_num_gpu >= job['fake_gpu_num']:
                    job['duration'] = job['fake_duration']
                    job['num_gpus'] = job['fake_gpu_num']
                    job['ps_network'] = [0.1 for _ in range(job['num_gpus'])]
                    assert job['num_gpus'] == job.required_gpu_num

                if total_num_gpu >= job['num_gpus']:
                    should_run_jobs.append(job)
                    total_num_gpu -= job['num_gpus']
                else:
                    break
        for job in self.pending_jobs:
            if total_num_gpu >= job['num_gpus'] and job not in should_run_jobs: 
                should_run_jobs.append(job)
                total_num_gpu -= job['num_gpus']
                
        return should_run_jobs, should_preempt_jobs


    def run(self, ):
        cur_time = 0
        while not self.finish_all_jobs():
            prev_time = max(0, cur_time - self.check_time_interval)
            self.update_job_emergence(cur_time)
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
        attribute_list = ['job_id', 'pending_time', 'total_executed_time', 'submit_time', 'end_time', 'preempt', 'resume', 'num_gpus', 'ddl_time', 'miss_ddl', 'ddl_time_list', 'ddl_value_list', 'ddl_type']
        
        for job in self.JOBS.job_list:
            job['miss_ddl'] = 0
            if job['ddl_type'] != 'best_effort': 
                job['miss_ddl'] = 1
                for vid, (expect_time, expect_value) in enumerate(zip(job['ddl_time_list'], job['ddl_value_list'])):
                    if expect_time + job['submit_time'] >= job['end_time']:
                        job['miss_ddl'] = 1 - expect_value * 1.0 / MAX_EXPECT_VALUE
                        break
            job['ddl_time_list'] =  '-'.join([str(item) for item in job['ddl_time_list']])
            job['ddl_value_list'] = '-'.join([str(item) for item in job['ddl_value_list']])
            
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