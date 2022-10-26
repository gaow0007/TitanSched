import math 
import numpy as np 
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict, cast
import math 
from client.application.foundation_model import FOUNDATIONMODELAPPLICATIONS
import collections 
from .base import BaseJob
from .state import JobState




class FoundationModelJob(BaseJob): 
    __alias__ = 'foundation_model' 
    def __init__(self, df, **kwargs): 
        super(FoundationModelJob, self).__init__(name=df.name, application=df.application, submission_time=df.submission_time)
        # speed information
        self.target_num_gpus = df.num_gpus
        
        self.application = FOUNDATIONMODELAPPLICATIONS[df.application]
        self.min_num_gpus = 1
        self.max_num_gpus = 32 # self.target_num_gpus * 4
        self.elastic = True 
        self.placement = None 
        self.preemptible = True 
        self.add_ckpt = kwargs.get('add_ckpt', 0)
        self.rescale_time = self.add_ckpt + self.application.get_context_switch_overhead(1, pipeline=False)

        # statistical information
        if hasattr(df, 'target_metric'): 
            self.target_gradient_steps = df.target_gradient_steps
            self.target_lr = df.target_lr
            self.target_metric = df.target_metric 
            # 1.936880 for 480 jobs
            self.max_progress = self.application.progress_per_epoch * \
                self.application.get_completion_epoch(lr=self.target_lr, gradient_steps=self.target_gradient_steps, target_metric=self.target_metric)
            # self.max_progress = self.application.progress_per_epoch * self.application.max_epochs
            # print('name {}, epoch {}'.format(self.name, self.max_progress // self.application.progress_per_epoch))
        else: 
            self.max_progress = self.application.progress_per_epoch * self.application.max_epochs
        self.target_batch_size = int(df.target_batch_size)
        self.max_num_gpus = min(self.max_num_gpus, self.target_batch_size // self.application.min_local_bsz)
         # * self.get_completion_epoch()
        self.atomic_bsz = 0 
        self.accum_steps = 0
        
        
        
    

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
            placcement = tuple([p for p in placement])
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
        if isinstance(placement, int): 
            num_gpus = placement
            placement = [4 for _ in range(num_gpus // 4)]
            if num_gpus % 4: placement.append(num_gpus % 4)
            placement = tuple(placement)
            assert sum(placement) == num_gpus
        elif isinstance(placement, collections.Iterable): 
            placcement = tuple([p for p in placement])
        else: 
            print(placement, type(placement))
            raise NotImplementedError
        
        self.update_local_bsz(placement)
        step_time, sync_time = self.application.get_throughput(placement, self.atomic_bsz)
        accum_time = step_time - sync_time
        total_time = step_time + accum_time * (self.accum_steps + 1)
        total_batch_size = sum(placement) * self.atomic_bsz * (self.accum_steps + 1)
        delta_progress = self.max_progress - self.progress
        delta_seconds = round(float(delta_progress / total_batch_size *  total_time)) 
        return delta_seconds


    def update_local_bsz(self, placement):
        app = self.application
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size 
        local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
        self.accum_steps = math.ceil(local_bsz / app.max_local_bsz - 1e-8) - 1
        self.atomic_bsz = math.ceil(local_bsz / (self.accum_steps + 1) - 1e-8)
        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = min(self.atomic_bsz, int(app.max_batch_size / count))

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
        if abs(self.progress - self.max_progress) > 0.1: 
            placement = tuple(filter(None, self.placement))
            self.update_local_bsz(placement) 
            
            step_time, sync_time = self.application.get_throughput(placement, self.atomic_bsz)
            accum_time = step_time - sync_time
            total_time = step_time + accum_time * self.accum_steps
            total_batch_size = sum(placement) * self.atomic_bsz * (self.accum_steps + 1)

            delta_progress = min(seconds / total_time * total_batch_size, self.max_progress - self.progress) 
            delta_seconds = round(float(delta_progress / total_batch_size * total_time)) 
            self.progress += delta_progress
            if abs(self.progress - self.max_progress) <= 0.1: 
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
        self.rescale_time = self.application.get_context_switch_overhead(1, pipeline=False)
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
                self.rescale_time = self.add_ckpt + self.application.get_context_switch_overhead(self.placement, pipeline=True)
            else: 
                self.rescale_time = self.add_ckpt + self.application.get_context_switch_overhead(self.placement, pipeline=False)

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