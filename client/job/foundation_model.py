import math 
import numpy as np 
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict, cast
import math 
from client import application
from client.application import FoundationModelApplication

from .base import BaseJob
from .state import JobState




class FoundationModelJob(BaseJob): 
    __alias__ = 'foundation_model' 
    def __init__(self, df): 
        super(FoundationModelJob, self).__init__(name=df.name, application=df.application, submission_time=df.submission_time)
        self.target_iteration = df.target_iteration 
        self.data_scale = df.data_scale 
        self.target_batch_size = df.target_batch_size
        self.target_iteration = df.target_iteration 
        self.application = FoundationModelApplication(df.application, self.data_scale, self.target_iteration)
        self.elastic = True 
        self.rescale_time = 0
        self.placement = None 
        self.preemptible = True 
    

    def step(self, seconds, **kwargs):
        if not self.placement:
            # No resources are allocated to this job.
            if self.completion_time is None: 
                self.staying_time += seconds
                self.status = JobState.PENDING
                self.pending_time += seconds 
            self.staying_time += seconds 
            return

        delay = min(self.rescale_time, seconds)
        self.rescale_time -= delay 
        seconds -= delay 

        self.status = JobState.RUNNING
        if abs(self.progress - self.application.max_progress) > 0.1: 
            placement = tuple(filter(None, self.placement))
            iteration_per_time = self.application.get_throughput(placement, self.target_batch_size)
            delta_progress = min(seconds * iteration_per_time, self.application.max_progress - self.progress) 
            delta_seconds = round(float(delta_progress / iteration_per_time)) 
            self.progress += delta_progress
            if abs(self.progress - self.application.max_progress) <= 0.1: 
                self.completion_time = self.staying_time + delta_seconds + self.submission_time 
            self.attained_service += delta_seconds * sum(placement) 
        
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
                self.num_restarts += 1
            kwargs['pipeline'] = False
            if kwargs.get('pipeline', False): 
                self.rescale_time = self.application.get_context_switch_overhead(self.placement, pipeline=True)
            else: 
                self.rescale_time = self.application.get_context_switch_overhead(self.placement, pipeline=False)

        else:  # De-allocate all resources.
            self.placement = None 

    def release_resource(self,): 
        self.placement = None 
        self.topology = None 
        self.status = JobState.PENDING



class MergeFoundationModelJob(BaseJob): 
    __alias__ = 'merge_foundation_model' 
    def __init__(self, fm_list: List[FoundationModelJob], submission_time: int): 
        assert len(fm_list) >= 2, 'at lease merge two `FM` tasks'
        super(MergeFoundationModelJob, self).__init__(name='-'.join([str(fm.name) for fm in fm_list]), application=fm_list[0].application.name, submission_time=submission_time)
        self.fm_list = fm_list
        merge_data_scale = sum([fm.application.data_scale for fm in fm_list])
        self.data_scale = merge_data_scale
        p0 = -6761
        p1 = 2327
        
        self.target_iteration = int(math.ceil(p0 + p1 * np.log(self.data_scale)))
        for fm_job in fm_list: 
            iteration  = int(math.ceil(p0 + p1 * np.log(fm_job.application.data_scale)))
            data_scale = fm_job.application.data_scale
            print('iteration {}, data scale {}, true iteration {}'.format(iteration, data_scale, fm_job.target_iteration))
        print('combined iteration {}, combined data scale {}'.format(self.target_iteration, self.data_scale))
        # import pdb; pdb.set_trace() 

        FM = fm_list[0]
        self.target_batch_size = FM.target_batch_size
        self.application = FoundationModelApplication(FM.application.name, self.data_scale, self.target_iteration)

        self.elastic = True 
        self.rescale_time = 0
        self.placement = None 
        self.preemptible = True 
    

    def step(self, seconds, **kwargs):
        if not self.placement:
            # No resources are allocated to this job.
            if self.completion_time is None: 
                self.staying_time += seconds
                self.status = JobState.PENDING
                self.pending_time += seconds 
            self.staying_time += seconds 
            return

        delay = min(self.rescale_time, seconds)
        self.rescale_time -= delay 
        seconds -= delay 

        self.status = JobState.RUNNING
        if self.progress < self.application.max_progress: 
            placement = tuple(filter(None, self.placement))
            iteration_per_time = self.application.get_throughput(placement, self.target_batch_size)
            delta_progress = min(seconds * iteration_per_time, self.application.max_progress - self.progress) 
            delta_seconds = round(float(delta_progress / iteration_per_time)) 
            self.progress += delta_progress
            if self.progress >= self.application.max_progress: 
                self.completion_time = self.staying_time + delta_seconds + self.submission_time 
            self.attained_service += delta_seconds * sum(placement) 
        
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
                self.num_restarts += 1
            kwargs['pipeline'] = False
            if kwargs.get('pipeline', False): 
                self.rescale_time = self.application.get_context_switch_overhead(self.placement, pipeline=True)
            else: 
                self.rescale_time = self.application.get_context_switch_overhead(self.placement, pipeline=False)

        else:  # De-allocate all resources.
            self.placement = ()

    def release_resource(self,): 
        self.placement = None 
        self.topology = None 
        self.status = JobState.PENDING
    
    def split_after_complete(self, ): 
        for fm_job in self.fm_list: 
            
            fm_job.attained_service = self.attained_service 
            fm_job.placement = self.placement 
            fm_job.status = self.status 

            fm_job.completion_time = self.completion_time
            fm_job.staying_time = fm_job.pending_time + self.staying_time 
            fm_job.running_time = self.running_time 
            fm_job.pending_time = fm_job.pending_time + self.pending_time
            fm_job.num_restarts = self.num_restarts 

        return self.fm_list