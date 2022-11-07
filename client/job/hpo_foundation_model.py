import math 
import numpy as np 
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict, cast
import math 
from client.application.foundation_model import FOUNDATIONMODELAPPLICATIONS
import collections 
from .base import BaseJob
from .state import JobState
import copy

DEFAULT_GPU_KIND="V100"
CRITICAL_EPOCH_POINT = [1, 3, 9, 100]

class HPOFoundationModelJob(BaseJob): 
    __alias__ = 'hpo_foundation_model' 
    def __init__(self, df, **kwargs): 
        super(HPOFoundationModelJob, self).__init__(name=df.name, application=df.application, submission_time=df.submission_time)
        # speed information
        self.target_num_gpus = df.num_gpus
        
        self.application = FOUNDATIONMODELAPPLICATIONS[df.application]
        self.min_num_gpus = 1
        self.max_num_gpus = 32 # self.target_num_gpus * 4
        self.elastic = True 
        self.placement = None 
        self.preemptible = True 
        self.add_ckpt = kwargs.get('add_ckpt', 0)
        self.rescale_time = self.add_ckpt + self.application.get_context_switch_overhead(GPU_KIND=DEFAULT_GPU_KIND, placement=1, pipeline=False)


        self.max_progress = self.application.progress_per_epoch * self.application.max_epochs
        self.training_epochs = self.application.max_epochs
        self.target_batch_size = int(df.target_batch_size)
        self.target_lr = df.target_lr 
        self.target_gradient_steps = df.target_gradient_steps
        self.max_num_gpus = min(self.max_num_gpus, self.target_batch_size // self.application.min_local_bsz)
        self.total_time_running_exclusively = self.predict_remaining_time(df.num_gpus)

        
        self.atomic_bsz = 0 
        self.accum_steps = 0
        self.physical = kwargs.get('physical', 'False')
        self.failure_ratio = kwargs.get('failure_ratio', 0.05)
        self.next_target_epoch = CRITICAL_EPOCH_POINT[0]
        self.next_critical_point = 0 
        self.cross_boundary = False 
        self.rank = -1 
        self.number_of_peers = -1 
        self.max_of_peers = -1 
        self.num_restarts = 0


    def get_current_epoch(self, ): 
        return self.progress / self.application.progress_per_epoch

    def get_actual_loss_value(self, predict_progress):
        normalized_progress = min(1, 1.0 * predict_progress / self.max_progress)
        return 1000. / (1 + 0.25 * normalized_progress)
    
    def predict_loss(self, placement, step_interval=30): 
        if isinstance(placement, int): 
            num_gpus = placement
            placement = [4 for _ in range(num_gpus // 4)]
            if num_gpus % 4: placement.append(num_gpus % 4)
            placement = tuple(placement)
            assert sum(placement) == num_gpus
        elif isinstance(placement, collections.Iterable): 
            placement = tuple([p for p in placement])
        else: 
            raise NotImplementedError
        
        if sum(placement) == 0: 
            predict_progress = self.progress 
        else:
            step_time = self.application.get_throughput(placement=placement, local_bsz=self.application.min_local_bsz) 
            if isinstance(step_time, (tuple, np.ndarray)): 
                step_time = step_time[0]
            throughput = 1. / step_time 
            predict_progress = self.progress + throughput * step_interval

        return self.get_actual_loss_value(predict_progress)

    def predict_remaining_time(self, placement): 
        GPU_KIND = DEFAULT_GPU_KIND
        if isinstance(placement, dict): 
            merge_placement = ()
            for item in placement.values(): 
                merge_placement += item 
            placement = merge_placement

        if isinstance(placement, int): 
            num_gpus = placement
            placement = [4 for _ in range(num_gpus // 4)]
            if num_gpus % 4: placement.append(num_gpus % 4)
            placement = tuple(placement)
            assert sum(placement) == num_gpus
        elif isinstance(placement, collections.Iterable): 
            placement = tuple([p for p in placement])
        else: 
            # print(placement, type(placement))
            raise NotImplementedError
        
        self.update_local_bsz(placement)
        if hasattr(self, 'topolocy') and self.topology is not None: 
            import pdb; pdb.set_trace() 
        else: 
            step_time, sync_time = self.application.get_throughput(DEFAULT_GPU_KIND, placement, self.atomic_bsz)
        accum_time = step_time - sync_time
        total_time = step_time + accum_time * (self.accum_steps + 1)
        total_batch_size = sum(placement) * self.atomic_bsz * (self.accum_steps + 1)
        delta_progress = self.max_progress - self.progress
        delta_seconds = round(float(delta_progress / total_batch_size *  total_time)) 
        return delta_seconds


    def update_local_bsz(self, placement):
        GPU_KIND = DEFAULT_GPU_KIND
        if isinstance(placement, dict): 
            merge_placement = ()
            for item in placement.values(): 
                merge_placement += item 
            placement = merge_placement

        app = self.application
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size 
        local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
        self.accum_steps = math.ceil(local_bsz / app.max_local_bsz - 1e-8) - 1
        if num_replicas == 1 and batch_size > app.max_local_bsz:
            self.accum_steps = max(1, self.accum_steps)
                
        self.atomic_bsz = math.ceil(local_bsz / (self.accum_steps + 1) - 1e-8)
        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = max(min(self.atomic_bsz, int(self.target_batch_size / count)), self.application.min_local_bsz)

    def current_device(self, ): 
        if not hasattr(self, 'topology') or self.topology is None or len(self.topology) == 0:  
            return DEFAULT_GPU_KIND
        else: 
            return self.topology[0]['gpu_kind']


    def step(self, seconds, **kwargs):
        if not self.placement:
            # No resources are allocated to this job.
            if self.completion_time is None: 
                self.staying_time += seconds
                self.status = JobState.PENDING
                self.pending_time += seconds 
            return

        GPU_KIND = DEFAULT_GPU_KIND
        if isinstance(self.placement, dict): 
            merge_placement = ()
            for item in self.placement.values(): 
                merge_placement += item 
            self.placement = merge_placement

        if self.physical and np.random.rand() > 1 - self.failure_ratio: 
            self.staying_time += seconds 
            self.running_time += seconds
            self.status = JobState.RUNNING
            return 
        self.status = JobState.RUNNING


        delay = min(self.rescale_time, seconds)
        self.rescale_time -= delay 
        seconds -= delay 
        self.staying_time += delay
        self.attained_service += delay * sum(self.placement)

        if abs(self.progress - self.max_progress) > 0.1: 
            placement = tuple(filter(None, self.placement))
            self.update_local_bsz(placement) 
            
            step_time, sync_time = self.application.get_throughput(GPU_KIND=self.current_device(), placement=placement, local_bsz=self.atomic_bsz)
            accum_time = step_time - sync_time
            total_time = step_time + accum_time * self.accum_steps
            total_batch_size = sum(placement) * self.atomic_bsz * (self.accum_steps + 1)

            delta_progress = min(seconds / total_time * total_batch_size, self.max_progress - self.progress) 
            delta_seconds = round(float(delta_progress / total_batch_size * total_time)) 
            self.progress += delta_progress
            if abs(self.progress - self.max_progress) <= 0.1: 
                self.completion_time = self.staying_time + delta_seconds + self.submission_time 
            self.attained_service += delta_seconds * sum(placement) 
        current_epoch = self.get_current_epoch()
        # while current_epoch >= CRITICAL_EPOCH_POINT[self.next_critical_point]: 
        #     self.next_critical_point += 1
        #     self.prepare_cross_boundary() 

        self.staying_time += delta_seconds 
        self.running_time += delta_seconds

    def early_stop(self, cur_time): 
        assert self.completion_time is None
        self.completion_time = cur_time 
    
    def prepare_cross_boundary(self, ): 
        self.cross_boundary = True 
    
    def process_cross_boundary(self, ): 
        self.cross_boundary = False
        

    def reallocate(self, placement, **kwargs):
        if placement:
            self.placement = tuple(placement)
            if kwargs.get('topology') is not None: 
                self.topology = kwargs.get('topology')
            else: 
                self.topology = None 

            if self.num_restarts is None:
                self.num_restarts = 1
            else:
                self.num_restarts += 1
            kwargs['pipeline'] = False
            GPU_KIND = self.current_device()
            
            if kwargs.get('pipeline', False): 
                self.rescale_time = self.add_ckpt + self.application.get_context_switch_overhead(GPU_KIND=GPU_KIND, placement=self.placement, pipeline=True)
            else: 
                self.rescale_time = self.add_ckpt + self.application.get_context_switch_overhead(GPU_KIND=GPU_KIND, placement=self.placement, pipeline=False)

        else:  # De-allocate all resources.
            self.placement = None 

    def release_resource(self,): 
        self.placement = None 
        self.topology = None 
        self.status = JobState.PENDING
    
    def query_metric(self, epoch): 
        return max(self.application.query_epoch(target_lr=self.target_lr, target_gradient_steps=self.target_gradient_steps, target_epoch=epoch), 1e-2) # should be more than 1e-2

    @property
    def reweight(self, ): 
        return 1
    
    
    @property
    def job_number(self, ): 
        return 1 
    
    @property
    def base_weight_scale(self, ):
        return 1

