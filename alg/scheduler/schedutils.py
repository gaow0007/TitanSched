import os, sys
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from client.job.state import JobState
import json

MAX_LEASE_NUM = 30 * 24 * 20 * 60
BE_REWARD = 25
SLO_REWARD = 1000
MAX_SEARCH_JOB = 10000


def hpo_schedule_summary(sched): 
    sched.logger.info('*' * 100)
    jct_info = dict() 
    avg_jct = 0 
    avg_service = 0 
    for application in sched.hpo_applications: 
        jct = 0 
        service = 0 
        jct_info[application] = dict() 
        jct_info[application]['jct'] = 0
        jct_info[application]['service'] = 0
        
        for job in sched.job_manager.job_list: 
            if application in job.application.name: 
                service += job.attained_service 
                if job.completion_time - job.submission_time > jct: 
                    jct = job.completion_time - job.submission_time
                
        jct_info[application]['jct'] = jct
        jct_info[application]['service'] = service
        avg_jct += jct / len(sched.hpo_applications)
        avg_service += service / len(sched.hpo_applications)
        sched.logger.info('HPO application {} jct {} hours'.format(application, jct / 3600))
        sched.logger.info('HPO application {} service {} GPU hours'.format(application, service / 3600))

    sched.logger.info('HPO average jct {} hours'.format(avg_jct / 3600))
    sched.logger.info('HPO average service {} GPU hours'.format(avg_service / 3600))

    with open(os.path.join(sched.save_dir, sched.name + '_hpo.json'), "w") as f:
        json.dump(jct_info, f)
        f.write("\n")
    sched.logger.info('*' * 100)


def schedule_summary(sched):
    for job in sched.job_manager.job_list: 
        if job.status != JobState.END: 
            import pdb; pdb.set_trace() 

    assert all([job.status == JobState.END for job in sched.job_manager.job_list])
    attribute_list = ['name', 'submission_time', 'pending_time', 'staying_time', 'completion_time', 'num_restarts', 'max_progress', 'target_num_gpus']
        
    with open(os.path.join(sched.save_dir, sched.name + '.csv'), 'w') as f:
        print(",".join(attribute_list), file=f)
        for job in sched.job_manager.job_list:
            for attribute in attribute_list:
                print("{}".format(getattr(job, attribute)), file=f) if attribute == attribute_list[-1] else print("{}".format(getattr(job, attribute)), end=',', file=f)
    
    attribute_list = ['full_resource', 'free_resource', 'pending_num', 'running_num', 'submit_num']
    with open(os.path.join(sched.save_dir, sched.name + '_resource.csv'), 'w') as f:
        print(",".join(attribute_list), file=f)
        for full_resource, free_resource, pending_num, running_num, submit_num in zip(sched.full_resource_list, sched.free_resource_list, sched.pending_job_num_list, sched.running_job_num_list, sched.submit_job_num_list):
            print('{},{},{},{},{}'.format(full_resource, free_resource, pending_num, running_num, submit_num), file=f)
    
    attribute_list = ['name', 'finish_time_fairness']
    with open(os.path.join(sched.save_dir, sched.name + '_fairness.csv'), 'w') as f: 
        print(",".join(attribute_list), file=f)
        for job in sched.job_manager.job_list: 
            completion_time = job.completion_time - job.submission_time
            total_time_running_exclusively = job.total_time_running_exclusively
            avg_contention_factor = max(1, len(sched.job_manager.job_list) / sched.cluster_manager.check_total_gpus())
            finish_time_fairness = round(
                completion_time / (total_time_running_exclusively * avg_contention_factor),
                5,
            )
            print('{},{}'.format(job.name, finish_time_fairness), file=f)
            job.finish_time_fairness = finish_time_fairness

    if hasattr(sched, 'records'): 
        with open(os.path.join(sched.save_dir, sched.name + '_logs.json'), "w") as f:
            for record in sched.records:
                json.dump(record, f)
                f.write("\n")

def resource_summary(sched, ):
    sched.full_resource_list.append(sched.cluster_manager.check_total_gpus())
    sched.free_resource_list.append(sched.cluster_manager.check_free_gpus())
    sched.pending_job_num_list.append(len(sched.pending_jobs))
    sched.running_job_num_list.append(len(sched.running_jobs))

def job_summary(sched, cur_time): 
    job_info = dict() 
    job_info['timestamp'] = cur_time 
    job_info['cluster']  = {
        'free' : sched.cluster_manager.check_free_gpus(), 
        'total': sched.cluster_manager.check_total_gpus(), 
    }
    job_info['job_dist'] = {
        'event' : len(sched.event_jobs), 
        'pending' : len(sched.pending_jobs), 
        'running' : len(sched.running_jobs), 
        'completion': len(sched.completion_jobs),
    }
    job_info['jobs'] = [
        {
            "name": job.name, 
            "placement": job.placement, 
            "progress": job.progress / job.max_progress, 
            "scale": job.application.scale,
            "progress_per_epoch": job.application.progress_per_epoch, 
        } 
        for job in sched.running_jobs
    ]
    sched.records.append(job_info)

class JobInfo(object):
    def __init__(self, resources, speedup_fn, creation_timestamp, attained_service,
                 min_replicas, max_replicas, preemptible=True, max_progress=None, name=None):
        """
        Args:
            resources (dict): Requested resources (eg. GPUs) of each replica.
            speedup_fn (SpeedupFunction): Speedup function for this job.
            creation_timestamp (datetime): Time when this job was created.
            min_replicas (int): Minimum number of replicas job's guaranteed.
            max_replicas (int): Maximum number of replicas. Maximum should be
                                greater or equal to Minimum
            preemptible (bool): Is the job preemptible?
        """
        assert max_replicas > 0
        assert max_replicas >= min_replicas
        self.resources = resources
        self.speedup_fn = speedup_fn
        self.creation_timestamp = creation_timestamp
        self.attained_service = attained_service
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas
        self.preemptible = preemptible
        self.max_progress = max_progress
        self.name = name 


class NodeInfo(object):
    def __init__(self, resources, preemptible):
        """
        Args:
            resources (dict): Available resources (eg. GPUs) on this node.
            preemptible (bool): Whether this node is pre-emptible.
        """
        self.resources = resources
        self.preemptible = preemptible