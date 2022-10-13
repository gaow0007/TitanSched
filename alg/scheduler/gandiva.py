
import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from .base import BaseScheduler
from utils.util import search_dict_list


class GandivaScheduler(BaseScheduler):
    def __init__(self, JOBS, CLUSTER, USERS, placement, name, logger, **kwargs):
        super(GandivaScheduler, self).__init__(JOBS=JOBS, CLUSTER=CLUSTER, USERS=USERS, placement=placement, name=name, logger=logger)
        assert self.name == 'gandiva'

        self.pending_jobs = JOBS.pending_jobs # list()
        self.running_jobs = JOBS.running_jobs
        self.runnable_jobs = JOBS.runnable_jobs # pending + running
        self.event_jobs = JOBS.job_events
        self.end_jobs = list()
        self.end_events = list() # collect on-going ending jobs
        self.check_time_interval = kwargs.get('check_time_interval', 10)
        self.save_dir = kwargs.get('save_dir', 'result/')


        
    def finish_all_jobs(self, ):
        return len(self.event_jobs) + len(self.pending_jobs) + len(self.running_jobs) == 0
    
    
    def try_allocate_resoure(self, job):
        if self.placement.name in ['random', 'consolidate', 'gandiva', 'local_search']:
            ret = self.placement.place_jobs(job)
        else:
            raise NotImplementedError
        return ret
        
        
    def release_job_resource(self, job, status='END'):
        if self.placement.name == 'gandiva':
            # ret = self.CLUSTER.release_gandiva_job_resource(job, status)
            raise NotImplementedError
        else:
            ret = self.CLUSTER.release_job_resource(job, status)
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
    

    def remove_from_pending(self, job, event_time):       
        job['status'] = 'RUNNING'
        job['start_time'] = event_time
        job['end_time'] = job['start_time'] + job['duration']
        job['pending_time'] = job['start_time'] - job['submit_time']
        job['last_check_time'] = event_time
        self.pending_jobs.remove(job)


    def remove_by_status(self, job, status):
        if status == 'EVENT':
            pass 
            # self.event_jobs.remove(job)
        elif status == 'PENDING':
            self.pending_jobs.remove(job)
        elif status == 'RUNNING':
            self.running_jobs.remove(job)
        elif status == 'END':
            self.end_jobs.remove(job)
        else:
            raise NotImplementedError
        
    
    def add_by_status(self, job, status):
        if status == 'EVENT':
            self.event_jobs.append(job)
        elif status == 'PENDING':
            self.pending_jobs.append(job)
        elif status == 'RUNNING':
            self.running_jobs.append(job)
        elif status == 'END':
            self.end_jobs.append(job)
        else:
            raise NotImplementedError


    def switch_job_status(self, job, prev=None, cur=None, cur_time=None):
        assert prev is not None and cur is not None
        self.remove_by_status(job, prev)
        self.add_by_status(job, cur)
    
   
    def flush_event_jobs(self, prev_time, cur_time):
        event_list = list()
        for event in self.event_jobs:
            if event['time'] <= cur_time:
                assert event['time'] >= prev_time
                event_list.append(event)

        submit_job_num = 0
        for event in event_list:
            for start_job in event['start_jobs']:
                start_job['last_check_time'] = cur_time
                start_job['pending_time'] = cur_time - start_job['submit_time']
                if self.try_allocate_resoure(start_job):
                    start_job['start_time'] = cur_time
                    start_job['status'] = 'RUNNING'
                    self.logger.info('----job[%d] starts at time[%d]' % (start_job['job_id'], cur_time))
                    self.running_jobs.append(start_job)
                else:
                    start_job['status'] = 'PENDING'
                    self.logger.info('----job[%d] pends at time[%d]' % (start_job['job_id'], cur_time))
                    self.pending_jobs.append(start_job)
                submit_job_num += 1
            self.event_jobs.remove(event)
        self.submit_job_num_list.append(submit_job_num)
    

    def flush_running_jobs(self, prev_time, cur_time):
        self.CLUSTER.gandiva_node_set_adjust(cur_time, self.JOBS)
        self.CLUSTER.time_slicing_execute(cur_time, self.JOBS, cur_time-prev_time)


    def flush_pending_jobs(self, prev_time, cur_time):
        for job in self.pending_jobs:
            if self.try_allocate_resoure(job):
                self.remove_from_pending(job, cur_time)
                self.running_jobs.append(job)
                # self.switch_job_status(p_job, prev='PENDING', cur='RUNNING', cur_time=cur_time)
                self.logger.info('----job[%d] starts from pending' % job['job_id'])


    def flush_jobs(self, prev_time, cur_time, status):
        if status == 'EVENT':
            self.flush_event_jobs(prev_time, cur_time)
        
        elif status == 'RUNNING':
            self.flush_running_jobs(prev_time, cur_time)
        
        elif status == 'PENDING':
            self.flush_pending_jobs(prev_time, cur_time)
        
        else:
            raise NotImplementedError


    def schedule_summary(self, ):
        assert all([job['status'] == 'END' for job in self.JOBS.job_list])
        attribute_list = ['job_id', 'pending_time', 'total_executed_time', 'submit_time', 'end_time', 'preempt', 'resume', 'num_gpu', 'expect_time', 'miss_ddl']
        for job in self.JOBS.job_list:
            job['miss_ddl'] = 0 if job['expect_time'] + job['submit_time'] >= job['end_time'] else 1
            
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


    def run(self, ):
        cur_time = 0

        while not self.finish_all_jobs():
            prev_time = max(0, cur_time - self.check_time_interval)
            time_diff = cur_time - prev_time
            self.flush_jobs(prev_time, cur_time, status='RUNNING')
            self.flush_jobs(prev_time, cur_time, status='EVENT') # first event jobs
            self.flush_jobs(prev_time, cur_time, status='PENDING')
            cur_time += self.check_time_interval
            self.resource_summary()
        
        self.schedule_summary()
        

