# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .base import BaseJob
from .state import JobState
from .elastic_utils import SpeedupFunction, GoodputFunction, fit_perf_params
from client.application import ResourceElasticApplication, BatchElasticApplication
from client.application.foundation_model import FOUNDATIONMODELAPPLICATIONS
import collections 
import math 
import numpy as np 

DEFAULT_GPU_KIND="V100"

class ResourceElasticJob(BaseJob): 
    __alias__ = 'resource_elastic' 
    def __init__(self, df): 
        super(ResourceElasticJob, self).__init__(name=df.name, application=df.application, submission_time=df.submission_time)
        self.preemptible = True 
        self.placement = None 

        if not hasattr(self, 'application') or self.application is None or isinstance(self.application, ResourceElasticApplication): 
            self.application = ResourceElasticApplication(name=self.name, duration=df.duration, num_gpus=df.num_gpus)

    def get_actual_loss_value(self, predict_progress):
        normalized_progress = min(1, 1.0 * predict_progress / self.application.max_progress)
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
            predict_progress = self.progress + self.application.get_throughput(placement) * step_interval

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
            raise NotImplementedError

        remaining_time = int((self.application.max_progress - self.progress) / self.application.get_throughput(placement))
        # print('remaining time {}, placement {}, max gpus limit {}'.format(remaining_time, placement, self.application.max_num_gpus))
        return remaining_time


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
        throughput = self.application.get_throughput(self.placement)
        if abs(self.progress - self.application.max_progress) > 0.1 : 
            delta_seconds = min(seconds, (self.application.max_progress - self.progress) / throughput)
            self.progress += delta_seconds * throughput
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
                assert 'base job should not be preepmted'
                self.num_restarts += 1

        else:  # De-allocate all resources.
            self.placement = None 
    
    def release_resource(self,): 
        self.placement = None 
        self.topology = None 
        self.status = JobState.PENDING


class BatchElasticJob(BaseJob): 
    __alias__ = 'batch_elastic' 
    def __init__(self, df, **kwargs): 
        super(BatchElasticJob, self).__init__(name=df.name, application=df.application, submission_time=df.submission_time)
        self.preemptible = True 
        if not hasattr(self, 'application') or self.application is None or isinstance(self.application, ResourceElasticApplication): 
            self.application = FOUNDATIONMODELAPPLICATIONS[df.application]

        self.target_batch_size = None # self.application.init_batch_size
        self.completion_time = None
        self.current_time = self.submission_time
        self.rescale_time = 0
        self.placement = None 
        self.atomic_bsz = 0
        self.accum_steps = 0
        self.profile = {}
        self.perf_params = None
        self.grad_params = None
        self.best_metric = None
        self.progress = 0.0
        self.epoch = 0
        self.attained_service = 0
        self.num_restarts = None
        
        self.add_ckpt = kwargs.get('add_ckpt', 0)
        self.rescale_time = self.add_ckpt + self.application.get_context_switch_overhead(GPU_KIND=DEFAULT_GPU_KIND, placement=1, pipeline=False)

        # statistical information
        if hasattr(df, 'target_metric'): 
            self.target_gradient_steps = df.target_gradient_steps
            self.target_lr = df.target_lr
            self.target_metric = float(df.target_metric)
            self.max_progress = self.application.progress_per_epoch * \
                self.application.get_completion_epoch(lr=self.target_lr, gradient_steps=self.target_gradient_steps, target_metric=self.target_metric)
            # self.max_progress = self.application.progress_per_epoch * self.application.max_epochs
            # print('name {}, epoch {}'.format(self.name, self.max_progress // self.application.progress_per_epoch))
            self.training_epochs = self.application.get_completion_epoch(lr=self.target_lr, gradient_steps=self.target_gradient_steps, target_metric=self.target_metric)
        else: 
            self.max_progress = self.application.progress_per_epoch * self.application.max_epochs
            self.training_epochs = self.application.max_epochs
        self.target_batch_size = int(df.target_batch_size)
        self.max_num_gpus = 32
        self.max_num_gpus = min(self.max_num_gpus, self.target_batch_size // self.application.min_local_bsz)
        
        self.atomic_bsz = 0 
        self.accum_steps = 0

        self.physical = kwargs.get('physical', 'False')
        self.failure_ratio = kwargs.get('failure_ratio', 0.05)


    @property
    def max_profiled_replicas(self):
        return max((k[1] for k in self.profile), default=0)

    def get_goodput_fn(self):
        app = self.application
        return GoodputFunction(self.perf_params, self.grad_params, self.target_batch_size)

    def get_speedup_fn(self):
        if self.perf_params is None:
            return lambda n, r: r
        app = self.application
        return SpeedupFunction(self.get_goodput_fn(), self.target_batch_size,
                               (app.min_local_bsz, app.max_local_bsz),
                               accumulation=True)

    def update_local_bsz(self, placement):
        app = self.application
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size
        if batch_size is None and self.perf_params is None:
            batch_size = max(app.init_batch_size, app.min_local_bsz * num_replicas)
        if batch_size is None:
            goodput_fn = self.get_goodput_fn()
            _, self.atomic_bsz, self.accum_steps = goodput_fn.optimize(
                num_nodes, num_replicas, self.target_batch_size,
                (app.min_local_bsz, app.max_local_bsz), accumulation=True)
        else:
            local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
            self.accum_steps = math.ceil(local_bsz / app.max_local_bsz - 1e-8) - 1
            if num_replicas == 1 and batch_size > app.max_local_bsz:
                self.accum_steps = max(1, self.accum_steps)
            self.atomic_bsz = math.ceil(local_bsz / (self.accum_steps + 1) - 1e-8)
        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = max(min(self.atomic_bsz, int(self.target_batch_size / count)), self.application.min_local_bsz)

    def reupdate(self, ): 
        num_nodes = np.array([key[0] for key in self.profile])
        num_replicas = np.array([key[1] for key in self.profile])
        local_bsz = np.array([key[2] for key in self.profile])
        step_time = np.array([val[0] for val in self.profile.values()])
        sync_time = np.array([val[1] for val in self.profile.values()])
        compute_time = step_time - sync_time
        self.perf_params = fit_perf_params(
            num_nodes, num_replicas, local_bsz, compute_time, step_time)
            
    def update_params(self, num_nodes, num_replicas, local_bsz,
                      step_time, sync_time, grad_sqr, grad_var):
        self.grad_params = (grad_sqr, grad_var)
        if (num_nodes, num_replicas, local_bsz) in self.profile:
            return
        self.profile[num_nodes, num_replicas, local_bsz] = step_time, sync_time
        num_nodes = np.array([key[0] for key in self.profile])
        num_replicas = np.array([key[1] for key in self.profile])
        local_bsz = np.array([key[2] for key in self.profile])
        step_time = np.array([val[0] for val in self.profile.values()])
        sync_time = np.array([val[1] for val in self.profile.values()])
        compute_time = step_time - sync_time
        self.perf_params = fit_perf_params(
            num_nodes, num_replicas, local_bsz, compute_time, step_time)

    def current_device(self, ): 
        if not hasattr(self, 'topology') or self.topology is None or len(self.topology) == 0:  
            return DEFAULT_GPU_KIND
        else: 
            return self.topology[0]['gpu_kind']
            
    def step(self, seconds, interference=0.0):
        if not self.placement:
            # No resources are allocated to this job.
            if self.completion_time is None: 
                self.staying_time += seconds
                self.status = JobState.PENDING
                self.pending_time += seconds 
            return

        if self.physical and np.random.rand() > 1 - self.failure_ratio: 
            self.staying_time += seconds 
            self.running_time += seconds
            self.status = JobState.RUNNING
            return 

        delay = min(self.rescale_time, seconds)
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.rescale_time -= delay
        seconds -= delay
        self.staying_time += delay
        self.status = JobState.RUNNING

        if abs(self.progress - self.max_progress) > 0.1 and seconds > 0: 
            assert self.epoch < self.application.max_epochs
            # Calculate current job configurations.
            placement = tuple(filter(None, self.placement))
            self.update_local_bsz(self.placement)
            num_nodes, num_replicas = len(placement), sum(placement)
            local_bsz = self.atomic_bsz
            batch_size = num_replicas * self.atomic_bsz * (self.accum_steps + 1)
            if self.target_batch_size is not None: 
                scale = 1.0
            else:
                scale = batch_size / self.application.init_batch_size
            # Calculate true (simulated) throughput.
            step_time, sync_time = \
                self.application.get_throughput(GPU_KIND=self.current_device(), placement=placement, local_bsz=self.atomic_bsz)
            accum_time = step_time - sync_time
            # Calculate true (simulated) efficiency.
            if self.target_batch_size is not None: 
                gain = 1.0
                grad_sqr = 1.0 
                grad_var = 1.0
            else: 
                grad_sqr, grad_var = \
                    self.application.get_grad_stats(batch_size, self.epoch)
                gain = (grad_var + grad_sqr) / (grad_var / scale + grad_sqr)

            # Update the estimated throughput/efficiency parameters.
            self.update_params(num_nodes, num_replicas, self.atomic_bsz,
                               step_time, sync_time, grad_sqr, grad_var)
            # Calculate true (simulated) goodput.
            total_time = step_time + accum_time * self.accum_steps

            # Update current epoch and progress.
            total_batch_size = sum(placement) * self.atomic_bsz * (self.accum_steps + 1)
            delta_progress = min(seconds / total_time * total_batch_size, self.max_progress - self.progress) 
            delta_seconds = round(float(delta_progress / total_batch_size * total_time)) 
            self.progress += delta_progress
            
            if abs(self.progress - self.max_progress) <= 0.1: 
                self.completion_time = self.staying_time + delta_seconds + self.submission_time 
            self.attained_service += delta_seconds * sum(placement) 
            
            
        self.staying_time += delta_seconds 
        self.running_time += delta_seconds
        self.current_time += delta_seconds

    def reallocate(self, placement, **kwargs):
        if placement:
            if sum(placement) == 0: 
                import pdb; pdb.set_trace() 
            self.placement = tuple([p for p in placement if p != 0])
            self.update_local_bsz(self.placement)

            if kwargs.get('topology') is not None: 
                self.topology = kwargs.get('topology')

            if self.num_restarts is None:
                self.num_restarts = 0
            else:
                assert 'base job should not be preepmted'
                self.num_restarts += 1
            self.rescale_time = self.add_ckpt + self.application.get_context_switch_overhead(GPU_KIND=DEFAULT_GPU_KIND, placement=1, pipeline=False)

        else:  # De-allocate all resources.
            self.placement = None 

    def release_resource(self,): 
        self.placement = None 
        self.topology = None 
        self.status = JobState.PENDING