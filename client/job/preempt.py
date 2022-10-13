import math 
from client.application import PhonyApplication
from .base import BaseJob
from .state import JobState

def get_context_switch_overhead(placement): 
    ckpt_multiplier = 1
    placement2context_switch_overhead_no_pipeline = {
            1 : 18*ckpt_multiplier,
            2: 22*ckpt_multiplier,
            3: 29*ckpt_multiplier,
            4: 29*ckpt_multiplier,
            5: 29*ckpt_multiplier,
            6: 29*ckpt_multiplier,
            7: 29*ckpt_multiplier,
            8: 46*ckpt_multiplier,
        }
    if sum(placement) in placement2context_switch_overhead_no_pipeline: 
        return placement2context_switch_overhead_no_pipeline[sum(placement)]
    else: 
        return sum(placement) / 8 * 46 * ckpt_multiplier


class PreemptJob(BaseJob): 
    __alias__ = 'preempt' 
    def __init__(self, df): 
        super(PreemptJob, self).__init__(name=df.name, application=df.application if hasattr(df, 'application') else None, submission_time=df.submission_time, 
                                        target_duration=df.duration, target_num_replicas=df.num_gpus, target_gpus_per_replica=1)
        self.preemptible = True 
        self.rescale_time = 0 
        if not hasattr(self, 'application') or self.application is None: 
            self.application = PhonyApplication(name=self.name, duration=self.target_duration)


    def step(self, seconds, **kwargs):
        if not self.placement:
            # No resources are allocated to this job.
            if self.completion_time is None: 
                self.staying_time += seconds
                self.pending_time += seconds
                self.status = JobState.PENDING
            return 

        delay = min(self.rescale_time, seconds)
        self.rescale_time -= delay 
        seconds -= delay 
        self.status = JobState.RUNNING
        
        if abs(self.progress - self.application.max_progress) > 0.1 : 
            delta_seconds = min(seconds, self.application.max_progress - self.progress)
            self.progress += delta_seconds
            if abs(self.progress - self.application.max_progress) <= 0.1: 
                self.completion_time = self.staying_time + delta_seconds + self.submission_time 
            self.attained_service += delta_seconds * sum(self.placement)
        try: 
            self.staying_time += delta_seconds 
            self.running_time += delta_seconds  
        except: 
            import pdb; pdb.set_trace() 


    def reallocate(self, placement, **kwargs):
        if placement:
            self.placement = tuple(placement)
            if kwargs.get('topology') is not None: 
                self.topology = kwargs.get('topology')
            if self.num_restarts is None:
                self.num_restarts = 0
            else:
                self.num_restarts += 1 
            if self.num_restarts == 0 or self.num_restarts is None: 
                self.rescale_time = 0
            else:
                self.rescale_time = get_context_switch_overhead(self.placement)
                
        else:  # De-allocate all resources.
            self.placement = None 

    
    def release_resource(self,): 
        self.placement = None 
        self.topology = None 
        self.status = JobState.PENDING