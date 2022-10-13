import math 
import os, sys
import numpy as np 
from abc import ABCMeta, abstractmethod
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict, cast
# sys.path.insert(0, os.path.basename(__file__) + os.sep + '..')
from client.application import PhonyApplication
from .state import JobState


class JobInfo(object): 
    def __init__(self, job): 
        self.name = job.name 
        self.submission_time = job.submission_time 
        self.rescale_time = job.rescale_time 
        self.attained_service = job.attained_service 


class BaseJob(metaclass=ABCMeta):
    __alias__ = 'base'
    def __init__(self, name, submission_time, application=None, target_duration=None, 
                 target_num_replicas=None, target_gpus_per_replica=None):
        self.name = name
        
        self.target_num_replicas = target_num_replicas
        self.target_gpus_per_replica = target_gpus_per_replica
        if self.target_num_replicas is not None and self.target_gpus_per_replica is not None: 
            self.target_num_gpus = self.target_num_replicas * self.target_gpus_per_replica
        else: 
            self.target_num_gpus = None 

        
        # time information 
        self.submission_time = submission_time
        self.completion_time = None
        self.rescale_time = 0
        self.pending_time = 0 
        self.running_time = 0 
        self.staying_time = 0 
        self.attained_service = 0

        # task specific information 
        if application is None: 
            self.application = PhonyApplication(name=name, duration=target_duration)
        
        self.target_duration = target_duration 

        self.epoch = 0
        self.progress = 0.0

        # schedule information
        self.num_restarts = None 
        self.status = JobState.EVENT
        self.placement = None 

        
    def step(self, seconds, **kwargs):
        if not self.placement:
            # No resources are allocated to this job.
            if self.completion_time is None: 
                self.staying_time += seconds
                self.status = JobState.PENDING
                self.pending_time += seconds 
            return
        
        self.status = JobState.RUNNING
        if abs(self.progress - self.application.max_progress) > 0.1 : 
            delta_seconds = min(seconds, self.application.max_progress - self.progress)
            self.progress += delta_seconds
            if abs(self.progress - self.application.max_progress) <= 0.1: 
                self.completion_time = self.staying_time + delta_seconds + self.submission_time

            self.attained_service += delta_seconds * sum(self.placement)

        self.staying_time += delta_seconds 
        self.running_time += delta_seconds


    def reallocate(self, placement, **kwargs):
        if placement:
            self.placement = tuple(placement)
            if kwargs.get('topology') is not None: 
                self.topology = kwargs.get('topology')

            if self.num_restarts is None:
                self.num_restarts = 0
            else:
                assert 'base job should not be preepmted'
                self.num_restarts += 1

        else:  # De-allocate all resources.
            self.placement = None 
    
    def release_resource(self,): 
        self.placement = None 
        self.topology = None 
        self.status = JobState.PENDING



class JobManager(object): 
    def __init__(self, ): 
        self.num_job = 0 
        self.job_list : List[BaseJob] = list() 
        self.event_jobs : List[BaseJob] = list() 
        self.pending_jobs = list() 
        self.running_jobs = list() 
    
    def submit_job(self, job: BaseJob): 
        self.event_jobs.append(job)
        self.num_job += 1 
        self.job_list.append(job)
    
    def prepare_job_start_events(self, ): 
        self.event_jobs.sort(key = lambda e:e.submission_time)


def JobFactory(name):
    if name == 'base': 
        return BaseJob

    for subclass in BaseJob.__subclasses__():
        if subclass.__alias__ == name:
            return subclass

    raise NotImplementedError