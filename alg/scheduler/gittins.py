import os, sys
import bisect
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')

from .base import BaseScheduler
from .disc_priority import DiscPrioirtyScheduler
from .tiresias import DlasScheduler
from utils.util import search_dict_list



def cal_rev_gittins_index(job_dist, service, ut_delta=None, ut_delta_list=None):
    """
    gittins_index = P / E
    return reverse of gittins_index
    # the definition is at last page
    # https://www.usenix.org/sites/default/files/conference/protected-files/nsdi19_slides_gu.pdf 
    """
    dist = job_dist['dist']
    if service > (job_dist['dist'][-1] - 1):
        return 0.
    else:
        idx = bisect.bisect_right(dist, service)
        
    max_r_gi = 0
    if ut_delta_list is not None:
        for ut_delta in ut_delta_list:
            next_service = service + ut_delta
            # next_service = ut_delta
            if next_service > (job_dist['dist'][-1] - 1):
                idx_delta = job_dist['num'] - 1
            else:
                idx_delta = bisect.bisect_right(dist, next_service)
                # idx_delta = next(x[0] for x in enumerate(dist) if x[1] > next_service)
            
            p = round((idx_delta - idx) * 1. / (job_dist['num'] - idx), 5)

            e_sum = sum(dist[idx:idx_delta]) + ut_delta * (job_dist['num'] - idx_delta)
            e = round(e_sum / (job_dist['num'] - idx), 5)
            r_gi = round(p * 1000000. / e, 4) 
            if r_gi > max_r_gi:
                max_r_gi = r_gi

    return max_r_gi



def prepare_job_dist(AllJobList=None):
    if AllJobList is None:
        import csv
        job_dist_file = os.path.join(os.getcwd(), 'data/yarn-gput1000.csv')
        fd = open(job_dist_file, 'r')
        reader = csv.DictReader(fd, delimiter=',')
        durations = list()
        for row in reader:
            durations.append(int(row['duration']))
        fd.close()
    else:
        durations = list()
        for job in AllJobList:
            durations.append(job.required_gpu_num * job['duration'])
    

    total_len = len(durations)
    durations.sort()
    
    print("     %s samples are learned" % total_len)
    job_dist = {
        'num': total_len, 
        'dist': durations + [sys.maxsize],
    }
    return job_dist


def get_gittins_index(service, job_dist):
    if service > job_dist['dist'][-2]:
        return 0.
    idx = next(x[0] for x in enumerate(job_dist['dist'])if x[1] > service)
    return job_dist['gittins'][idx]



class GittinsScheduler(DiscPrioirtyScheduler):
    def __init__(self, JOBS, CLUSTER, USERS, placement, name, logger, **kwargs):
        super(GittinsScheduler, self).__init__(JOBS=JOBS, CLUSTER=CLUSTER, USERS=USERS, placement=placement, name=name, logger=logger)
        assert self.name == 'gittins'

        self.pending_jobs = JOBS.pending_jobs
        self.running_jobs = JOBS.running_jobs
        self.runnable_jobs = JOBS.runnable_jobs # pending + running
        self.event_jobs = JOBS.job_events
        self.end_jobs = list()
        self.end_events = list() # collect on-going ending jobs
        self.check_time_interval = kwargs.get('check_time_interval', 10)
        self.save_dir = kwargs.get('save_dir', 'result/')
    
        # gittins specified
        self.solve_starvation = kwargs.get('solve_starvation', 0) # supports avoiding starvation method
        self.set_queues(kwargs.get('num_queue'), kwargs.get('queue_limit'))
        self.job_distribution = prepare_job_dist(JOBS.job_list)



    # abstract method
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
        for job in jobs:
            if not self.try_allocate_resoure(job): # TODO, make it allowed allocate across nodes
                continue

            job['resume'] += 1
            job['status'] = 'RUNNING'
            if job['start_time'] == sys.maxsize:
                job['start_time'] = cur_time
            
            self.logger.info('pending job {} is resumed at time {}'.format(job['job_id'], cur_time))
    
    
    def update_job_gittins(self):
        # deserved: min {share, demand}
        for job in self.running_jobs:
            attianed_service = job['executed_time'] * job.required_gpu_num
            job['rank'] = cal_rev_gittins_index(job_dist=self.job_distribution, service=attianed_service, \
                ut_delta_list=[it * self.check_time_interval * job.required_gpu_num for it in range(1, 100)] ) # self.queue_limit[0])

            

    def reallocate_resource(self, total_num_gpu):
        should_run_jobs, should_preempt_jobs = list(), list()
        for idx, queue in enumerate(self.queues):
            assert all([x.__getitem__('queue_id') == idx for x in queue])
            queue.sort(key=lambda x: x.__getitem__('rank'), reverse=True)
            for job in queue:
                if total_num_gpu >= job['num_gpu']:
                    if job['status'] == 'PENDING':
                        should_run_jobs.append(job)
                    if job['status'] == 'PENDING' or job['status'] == 'RUNNING':
                        total_num_gpu -= job['num_gpu']
                else:
                    if job['status'] == 'RUNNING':
                        should_preempt_jobs.append(job)

        return should_run_jobs, should_preempt_jobs

    
    def flush_running_jobs(self, prev_time, cur_time):
        for job in self.running_jobs:
            time_diff = int(cur_time - job['last_check_time'])
            if time_diff + job['total_executed_time'] >= job['duration']:
                assert self.release_job_resource(job) == True
                job['end_time'] = job['last_check_time'] + job['duration'] - job['total_executed_time']
                self.update_job_time(job, job['end_time'], name='RUNNING')
                self.logger.info('complete job {} at time {}'.format(job['job_id'], job['end_time']))
                job['status'] = 'END'
                self.running_jobs.remove(job)
                self.end_jobs.append(job)
                self.queues[job['queue_id']].remove(job)
            else:
                self.update_job_time(job, cur_time, name='RUNNING')
                attianed_service = job['executed_time'] * job.required_gpu_num
                # check demotion or promotion in queue
                self.update_job_queue(job, service=attianed_service, name='service')
       

    def flush_event_jobs(self, prev_time, cur_time):
        event_list = list()
        for event in self.event_jobs:
            if event['time'] <= cur_time:
                assert event['time'] >= prev_time
                event_list.append(event)

        for event in event_list:
            for start_job in event['start_jobs']:
                self.execute_start_job(start_job, cur_time)
                start_job['last_check_time'] = cur_time
                start_job['pending_time'] = cur_time - start_job['submit_time']
                start_job['rank'] = cal_rev_gittins_index(job_dist=self.job_distribution, service=0, 
                ut_delta_list=[it * self.check_time_interval * start_job.required_gpu_num for it in range(1, 2)] ) 
            self.event_jobs.remove(event)
    
    def schedule_summary(self, ):
        assert all([job['status'] == 'END' for job in self.JOBS.job_list])
        attribute_list = ['job_id', 'pending_time', 'total_executed_time', 'submit_time', 'end_time', 'preempt', 'resume', 'num_gpu', 'miss_ddl']
        for job in self.JOBS.job_list:
            job['miss_ddl'] = 0 if job['expect_time'] + job['submit_time'] >= job['end_time'] else 1
            
        with open(os.path.join(self.save_dir, self.name + '.csv'), 'w') as f:
            print(",".join(attribute_list), file=f)
            for job in self.JOBS.job_list:
                for attribute in attribute_list:
                    print("{}".format(job[attribute]), file=f) if attribute == attribute_list[-1] else print("{}".format(job[attribute]), end=',', file=f)
        
        attribute_list = ['full_resource', 'free_resource']
        with open(os.path.join(self.save_dir, self.name + '_resource.csv'), 'w') as f:
            print(",".join(attribute_list), file=f)
            for full_resource, free_resource in zip(self.full_resource_list, self.free_resource_list):
                print('{},{}'.format(full_resource, free_resource), file=f)


    def resource_summary(self, ):
        self.full_resource_list.append(self.CLUSTER.check_total_gpus())
        self.free_resource_list.append(self.CLUSTER.check_free_gpus())


    def run(self, ):
        cur_time = 0
        while not self.finish_all_jobs():
            prev_time = max(0, cur_time - self.check_time_interval)
            self.flush_jobs(prev_time, cur_time, status='RUNNING')
            self.flush_jobs(prev_time, cur_time, status='EVENT')
            self.flush_jobs(prev_time, cur_time, status='PENDING')

            self.update_job_gittins()
            self.flush_jobs(prev_time, cur_time, status='RUNNABLE')
            cur_time += self.check_time_interval
            self.resource_summary()
        
        self.schedule_summary()
        