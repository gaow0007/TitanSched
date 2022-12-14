import math 
import numpy as np 
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict, cast
import math 
from client.application.foundation_model import FOUNDATIONMODELAPPLICATIONS
import collections 
from .base import BaseJob
from .state import JobState
import copy

HETER_DEBUG=False
DEFAULT_GPU_KIND="V100"

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
        
        self.max_num_gpus = min(self.max_num_gpus, self.target_batch_size // self.application.min_local_bsz)
        
        self.atomic_bsz = 0 
        self.accum_steps = 0
        self.physical = kwargs.get('physical', 'False')
        self.failure_ratio = kwargs.get('failure_ratio', 0.05)
        self.next_target_epoch = 0 
        self.estimation_error = kwargs.get('estimation_error', 0)
        # key step 
        tmp_error = self.estimation_error
        self.estimation_error = 0
        self.total_time_running_exclusively = self.predict_remaining_time(df.num_gpus)
        self.estimation_error = tmp_error 
        self.profile = kwargs.get('profile', False) 
        if self.profile: 
            self.max_num_gpus = min(self.max_num_gpus, 2)
        
    def set_max_gpus(self, cur_gpus):
        self.max_num_gpus = max(cur_gpus * 2, self.max_num_gpus)
        self.max_num_gpus = min(self.max_num_gpus, 32)
        self.max_num_gpus = min(self.max_num_gpus, self.target_batch_size // self.application.min_local_bsz)

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

    def heter_predict_remaining_time(self, placement): 
        assert isinstance(placement, dict) 
        self.update_local_bsz(placement)
        
        sync_time, accum_time = 0, 0
        merge_placement = () 
        total_batch_size = 0
        for GPU_KIND in ["V100", "A100"]: 
            single_placement = placement[GPU_KIND]
            merge_placement += single_placement
            
        for GPU_KIND in ["V100", "A100"]: 
            single_placement = placement[GPU_KIND]
            merge_placement += single_placement
            single_step_time, single_sync_time = self.application.get_throughput(GPU_KIND=GPU_KIND, placement=single_placement, local_bsz=self.atomic_bsz[GPU_KIND])
            accum_time = max(single_step_time - single_sync_time, accum_time)
            single_step_time, single_sync_time = self.application.get_throughput(GPU_KIND=GPU_KIND, placement=merge_placement, local_bsz=self.atomic_bsz[GPU_KIND])
            sync_time = max(single_sync_time, sync_time)
            total_batch_size += sum(single_placement) * self.atomic_bsz[GPU_KIND] * (self.accum_steps + 1)

        total_time = sync_time + accum_time * (self.accum_steps + 1)
        delta_progress = self.max_progress - self.progress
        delta_seconds = round(float(delta_progress / total_batch_size *  total_time)) 
        return delta_seconds
    

    def predict_remaining_time(self, placement): 
        GPU_KIND = DEFAULT_GPU_KIND
        # if isinstance(placement, dict) and HETER_DEBUG: 
        #     merge_placement = ()
        #     for item in placement.values(): 
        #         merge_placement += item 
        #     placement = merge_placement

        if isinstance(placement, dict) and len(placement) == 1: 
            GPU_KIND = list(placement.keys())[0]
            placement = list(placement.values())[0]

        if isinstance(placement, int): 
            num_gpus = placement
            placement = [4 for _ in range(num_gpus // 4)]
            if num_gpus % 4: placement.append(num_gpus % 4)
            placement = tuple(placement)
            assert sum(placement) == num_gpus

        elif isinstance(placement, dict): 
            return self.heter_predict_remaining_time(placement)

        elif isinstance(placement, collections.Iterable): 
            placement = tuple([p for p in placement])
        else: 
            # print(placement, type(placement))
            raise NotImplementedError
        
        self.update_local_bsz(placement)
        if hasattr(self, 'topolocy') and self.topology is not None: 
            import pdb; pdb.set_trace() 
        else: 
            # GPU_KIND = DEFAULT_GPU_KIND
            step_time, sync_time = self.application.get_throughput(GPU_KIND, placement, self.atomic_bsz)
        
        accum_time = step_time - sync_time
        total_time = step_time + accum_time * (self.accum_steps + 1)
        total_batch_size = sum(placement) * self.atomic_bsz * (self.accum_steps + 1)
        delta_progress = self.max_progress - self.progress
        delta_seconds = round(float(delta_progress / total_batch_size *  total_time)) 
        return delta_seconds * max((1 + self.estimation_error * np.random.randn()), 0.1)

    def compute_local_bsz(self, placement): 
        app = self.application
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size 
        local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
        accum_steps = math.ceil(local_bsz / app.max_local_bsz - 1e-8) - 1
        if num_replicas == 1 and batch_size > app.max_local_bsz:
            accum_steps = max(1, accum_steps)
                
        atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
        count = num_replicas * (accum_steps + 1)
        atomic_bsz = max(min(atomic_bsz, int(self.target_batch_size / count)), self.application.min_local_bsz)
        return atomic_bsz


    def heter_update_local_bsz(self, placement_dict):
        app = self.application
        num_nodes, num_replicas = 0, 0
        equivalent_num_replicas = 0 

        merge_placement = ()
        for gpu_kind, placement_item in placement_dict.items(): 
            merge_placement += placement_item

        
        if False: 
            atomic_bsz = self.compute_local_bsz(merge_placement)
            step_time_V100, sync_time_V100 = self.application.get_throughput(GPU_KIND="V100", placement=(1,), local_bsz=atomic_bsz)
            step_time_A100, sync_time_A100 = self.application.get_throughput(GPU_KIND="A100", placement=(1,), local_bsz=atomic_bsz)
            heter_ratio = step_time_V100 / step_time_A100

            for gpu_kind, placement_item in placement_dict.items(): 
                num_nodes += len(placement_item)
                num_replicas += sum(placement_item) 
                equivalent_num_replicas += sum(placement_item) * heter_ratio
            
            batch_size = self.target_batch_size
            local_bsz = int(math.ceil(batch_size / equivalent_num_replicas - 1e-8))
            
            self.accum_steps = math.ceil(local_bsz / app.max_local_bsz - 1e-8) - 1
            if num_replicas == 1 and batch_size > app.max_local_bsz:
                self.accum_steps = max(1, self.accum_steps)
                    
            atomic_bsz = math.ceil(local_bsz / (self.accum_steps + 1) - 1e-8)
            count = num_replicas * (self.accum_steps + 1)
            atomic_bsz = max(min(atomic_bsz, int(self.target_batch_size / count)), self.application.min_local_bsz)
            self.atomic_bsz = dict() 
            self.atomic_bsz["V100"] = atomic_bsz
            self.atomic_bsz["A100"] = max(min(int(atomic_bsz * heter_ratio), int(self.target_batch_size / count)), self.application.min_local_bsz)
        else: 
            V100_num_replicas = sum(placement_dict['V100'])
            A100_num_replicas = sum(placement_dict['A100'])

            best_V100_atomic_bsz = None 
            best_A100_atomic_bsz = None
            best_accum_steps = None 
            
            V100_atomic_bsz_list = sorted(self.application.fm_speed.heter_atomic_bsz.keys())
            for accum_steps in range(10): 
                max_V100_atomic_bsz = max(V100_atomic_bsz_list)
                max_A100_atomic_bsz = self.application.fm_speed.heter_atomic_bsz[max_V100_atomic_bsz]
                if (max_V100_atomic_bsz * V100_num_replicas + max_A100_atomic_bsz * A100_num_replicas) * (accum_steps + 1) < self.target_batch_size: 
                    continue 
                local_bsz = self.target_batch_size // (accum_steps + 1)
                diff = 1000 
                
                for V100_atomic_bsz in V100_atomic_bsz_list: 
                    A100_atomic_bsz = self.application.fm_speed.heter_atomic_bsz[V100_atomic_bsz]
                    total_batch_size = (A100_atomic_bsz * A100_num_replicas + V100_atomic_bsz * V100_num_replicas) * (accum_steps + 1)
                    if abs(total_batch_size - self.target_batch_size) < diff :
                        diff = abs(total_batch_size - self.target_batch_size)
                        best_V100_atomic_bsz = V100_atomic_bsz
                        best_A100_atomic_bsz = A100_atomic_bsz
                        best_accum_steps = accum_steps
                    if total_batch_size > self.target_batch_size: 
                        break 
                break 
            assert best_V100_atomic_bsz is not None
            self.atomic_bsz = {
                "V100": best_V100_atomic_bsz, 
                "A100": best_A100_atomic_bsz, 
            }
            self.accum_steps = best_accum_steps
            
            
    def update_local_bsz(self, placement):
        # print('placement == {}'.format(placement))
        GPU_KIND = DEFAULT_GPU_KIND
        # if isinstance(placement, dict) and HETER_DEBUG: 
        #     merge_placement = ()
        #     for item in placement.values(): 
        #         merge_placement += item 
        #     placement = merge_placement

        if isinstance(placement, dict) and len(placement) == 1: 
            GPU_KIND = list(placement.keys())[0]
            placement = list(placement.values())[0]

        if isinstance(placement, dict): 
            assert len(placement) == 2, 'we only consider two kinds of GPUs'
            for gpu_kind, placement_item in placement.items():
                assert sum(placement_item) != 0, 'we consider actual hetergeneous training'
            self.heter_update_local_bsz(placement)
            return 

        app = self.application
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        if num_replicas == 0: 
            import pdb; pdb.set_trace() 
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
        elif 'V100' in self.topology:
            return 'V100'
        elif 'A100' in self.topology: 
            return 'A100'
        else:  
            for topo in self.topology: 
                if topo['gpu_kind'] == 'V100': return 'V100'
            return 'A100'

    def heter_step(self, seconds, **kwargs): 
        assert isinstance(self.placement, dict)
        if abs(self.progress - self.max_progress) > 0.1: 
            self.update_local_bsz(self.placement) 
            accum_time, sync_time = 0, 0
            merge_placement = ()
            total_batch_size = 0
            for GPU_KIND in ["V100", "A100"]: 
                single_placement = self.placement[GPU_KIND]
                merge_placement += single_placement
            
            for GPU_KIND in ["V100", "A100"]: 
                single_placement = self.placement[GPU_KIND]
                single_step_time, single_sync_time = self.application.get_throughput(GPU_KIND=GPU_KIND, placement=single_placement, local_bsz=self.atomic_bsz[GPU_KIND])
                accum_time = max(single_step_time - single_sync_time, accum_time)
                _, single_sync_time = self.application.get_throughput(GPU_KIND=GPU_KIND, placement=merge_placement, local_bsz=self.atomic_bsz[GPU_KIND])
                sync_time = max(single_sync_time, sync_time)
                total_batch_size += sum(single_placement) * self.atomic_bsz[GPU_KIND] * (self.accum_steps + 1)
            
            # accum_time = step_time - sync_time
            total_time = sync_time + accum_time * (self.accum_steps + 1)
            if HETER_DEBUG: 
                for GPU_KIND in ["V100", "A100"]: 
                    single_step_time, single_sync_time = self.application.get_throughput(GPU_KIND=GPU_KIND, placement=merge_placement, local_bsz=self.atomic_bsz[GPU_KIND])
                    single_total_time = single_sync_time + (single_step_time - single_sync_time) * (self.accum_steps + 1)
                    print('gpu kind {} single total {} , true total {}'.format(GPU_KIND, single_total_time, total_time))
                    print('gpu kind {} single step time {}, single sync time {}'.format(GPU_KIND, single_step_time, single_sync_time))
                    # single_step_time, single_sync_time = self.application.get_throughput(GPU_KIND=GPU_KIND, placement=merge_placement, local_bsz)
                print(self.atomic_bsz)
                # total_time = single_total_time
                import pdb; pdb.set_trace() 

            delta_progress = min(seconds / total_time * total_batch_size, self.max_progress - self.progress) 
            delta_seconds = round(float(delta_progress / total_batch_size * total_time)) 
            self.progress += delta_progress
            if abs(self.progress - self.max_progress) <= 0.1: 
                self.completion_time = self.staying_time + delta_seconds + self.submission_time 
            self.attained_service += delta_seconds * sum(merge_placement) 
        
        self.staying_time += delta_seconds 
        self.running_time += delta_seconds

    
    def step(self, seconds, **kwargs):
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
        self.rescale_time -= delay 
        seconds -= delay 
        self.staying_time += delay
        self.status = JobState.RUNNING

        if isinstance(self.placement, dict): 
            if len(self.placement) == 2: 
                self.heter_step(seconds, **kwargs)
                return 
            else: 
                self.placement = list(self.placement.values())[0]

        self.attained_service += delay * sum(self.placement)
        if self.profile: 
            self.set_max_gpus(sum(self.placement))

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
        
        self.staying_time += delta_seconds 
        self.running_time += delta_seconds


    def reallocate(self, placement, **kwargs):
        if placement:
            if isinstance(placement, dict): 
                self.placement = {key:tuple(value) for key, value in placement.items()}
            else: 
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

    @property
    def reweight(self, ): 
        return 1
    
    
    @property
    def job_number(self, ): 
        return 1 
    
    @property
    def base_weight_scale(self, ):
        return 1



class TransferFoundationModelJob(FoundationModelJob): 
    __alias__ = 'transfer_foundation_model' 
    def __init__(self, modelA, modelB, **kwargs): 
        self.modelA = modelA
        self.modelB = modelB 
        for attr in dir(self.modelB): 
            if not attr.startswith('__') and not hasattr(self, attr): 
                if attr in ['reweight', 'job_number', 'base_weight_scale']: continue
                instance = getattr(self.modelB, attr)
                setattr(self, attr, copy.deepcopy(instance))

        self.name = 'transfer_{}_{}'.format(self.modelA.name, self.modelB.name)
        self.status = JobState.PENDING
        self.max_progress = self.modelB.application.progress_per_epoch * \
                self.modelB.application.get_completion_epoch(transfer=True, taskA=self.modelA.application.task_name, 
                                                        taskB=self.modelB.application.task_name, target_metric=self.modelB.target_metric)


    def split_after_complete(self, ): 
        for fm_job in [self.modelB]: 
            
            fm_job.attained_service = self.attained_service 
            fm_job.placement = self.placement 
            fm_job.status = self.status 

            fm_job.completion_time = self.completion_time
            fm_job.staying_time = fm_job.pending_time + self.staying_time 
            fm_job.running_time = self.running_time 
            fm_job.pending_time = fm_job.pending_time + self.pending_time
            fm_job.num_restarts = self.num_restarts 

        return self.modelB

    @property
    def reweight(self, ): 
        return self.modelB.max_progress / self.max_progress 
    
    
    @property
    def job_number(self, ): 
        return 1 
    
    @property
    def base_weight_scale(self, ):
        return 1


class TemporalTransferFoundationModelJob(FoundationModelJob): 
    __alias__ = 'temporal_transfer_foundation_model' 
    def __init__(self, modelA, modelB, **kwargs): 
        self.modelA = modelA
        self.modelB = modelB 
        for attr in dir(self.modelA): 
            if not attr.startswith('__') and not hasattr(self, attr): 
                if attr in ['reweight', 'job_number', 'base_weight_scale', 'step']: continue
                instance = getattr(self.modelA, attr)
                setattr(self, attr, copy.deepcopy(instance))

        self.name = 'temporal_transfer_{}_{}'.format(self.modelA.name, self.modelB.name)
        self.status = JobState.PENDING
        self.max_progress = self.modelA.max_progress +  self.modelB.application.progress_per_epoch * \
                self.modelB.application.get_completion_epoch(transfer=True, taskA=self.modelA.application.task_name, 
                                                        taskB=self.modelB.application.task_name, target_metric=self.target_metric)
        self.middle_max_progress = self.modelA.max_progress
        self.middle_completion_time = None 
        self.srtf_progress = self.modelA.max_progress + self.modelB.max_progress + min(self.modelA.max_progress, self.modelB.max_progress)


    def step(self, seconds, **kwargs):
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
        self.attained_service += delay * sum(placement)
        self.rescale_time -= delay 
        seconds -= delay 
        self.staying_time += delay
        self.status = JobState.RUNNING

        if self.profile: 
            self.set_max_gpus(sum(self.placement))
            
        if self.progress < self.middle_max_progress and self.middle_completion_time is None: 
            placement = tuple(filter(None, self.placement))
            self.update_local_bsz(placement) 
            
            step_time, sync_time = self.application.get_throughput(placement, self.atomic_bsz)
            accum_time = step_time - sync_time
            total_time = step_time + accum_time * self.accum_steps
            total_batch_size = sum(placement) * self.atomic_bsz * (self.accum_steps + 1)

            delta_progress = min(seconds / total_time * total_batch_size, self.max_progress - self.progress) 
            delta_seconds = round(float(delta_progress / total_batch_size * total_time)) 
            self.progress += delta_progress
            if abs(self.progress - self.middle_max_progress) <= 0.1: 
                self.middle_completion_time = self.staying_time + delta_seconds + self.submission_time 
                self.target_batch_size = self.modelB.target_batch_size

            self.attained_service += delta_seconds * sum(placement) 
            if self.middle_completion_time is not None: 
                self.staying_time += delta_seconds 
                self.running_time += delta_seconds
                self.register_complete_job(self.modelA)
            seconds -= delta_seconds

        if abs(self.progress - self.max_progress) > 0.1 and seconds > 0 and self.middle_completion_time is not None: 
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

            if self.middle_completion_time is not None: 
                self.staying_time += delta_seconds 
                self.running_time += delta_seconds
                self.register_complete_job(self.modelB)

            self.attained_service += delta_seconds * sum(placement) 
        
        self.staying_time += delta_seconds 
        self.running_time += delta_seconds


    def register_complete_job(self, fm_job, completion_time): 
        fm_job.attained_service = self.attained_service 
        fm_job.placement = self.placement 
        fm_job.status = self.status 

        fm_job.completion_time = completion_time - self.submission_time + fm_job.submission_time
        fm_job.staying_time = fm_job.pending_time + self.staying_time 
        fm_job.running_time = self.running_time 
        fm_job.pending_time = fm_job.pending_time + self.pending_time
        fm_job.num_restarts = self.num_restarts 

        # clean 
        self.attained_service = 0
        self.running_time = 0 
        self.pending_time = self.staying_time
        self.num_restarts = 0 


    def split_after_complete(self, ): 
        return [self.modelA, self.modelB]

    @property
    def reweight(self, ): 
        return self.srtf_progress / self.max_progress 
    
    
    @property
    def job_number(self, ): 
        return 1 
    
    @property
    def base_weight_scale(self, ):
        if self.middle_completion_time is None: 
            return 2 
        elif self.completion_time is None: 
            return 1 
        return 0


class MtaskFoundationModelJob(FoundationModelJob): 
    __alias__ = 'mtask_foundation_model' 
    def __init__(self, modelA, modelB, **kwargs): 
        assert modelA.application.query_index() != modelB.application.query_index(), 'both should not equal'
        if modelA.application.query_index() < modelB.application.query_index(): 
            self.modelA, self.modelB = modelA, modelB 
        else: 
            self.modelA, self.modelB = modelB, modelA

        for attr in dir(self.modelA): 
            if not attr.startswith('__') and not hasattr(self, attr): 
                if attr in ['reweight', 'job_number', 'base_weight_scale']: continue
                instance = getattr(self.modelA, attr)
                setattr(self, attr, copy.deepcopy(instance))
        
        self.name = 'mtask_{}_{}'.format(self.modelA.name, self.modelB.name)
        self.placement = None 
        self.status = JobState.PENDING
        self.target_metric = (self.modelA.target_metric, self.modelB.target_metric)
        self.max_num_gpus = self.modelA.max_num_gpus + self.modelB.max_num_gpus 
        self.target_num_gpus = None
        progress_per_epoch = (self.modelA.application.progress_per_epoch + self.modelB.application.progress_per_epoch)
        epochA = self.modelA.application.get_completion_epoch(mtask=True, taskA=self.modelA.application.task_name, 
                                                        taskB=self.modelB.application.task_name, target_metric=self.modelA.target_metric) 
        epochB = self.modelB.application.get_completion_epoch(mtask=True, taskA=self.modelA.application.task_name, 
                                                        taskB=self.modelB.application.task_name, target_metric=self.modelB.target_metric) 
        self.training_epochs = max(epochA, epochB)
        # self.max_progress = self.training_epochs * progress_per_epoch
        # self.max_progress = self.modelA.application.progress_per_epoch * epochA + self.modelB.application.progress_per_epoch * epochB
        self.srtf_progress = self.modelA.max_progress + self.modelB.max_progress + min(self.modelA.max_progress, self.modelB.max_progress)
        self.target_batch_size = self.modelA.target_batch_size + self.modelB.target_batch_size
        self.atomic_bsz = 0 
        self.accum_steps = 0
        

    def split_after_complete(self, ): 
        for fm_job in [self.modelA, self.modelB]: 
            
            fm_job.attained_service = self.attained_service 
            fm_job.placement = self.placement 
            fm_job.status = self.status 

            fm_job.completion_time = self.completion_time - self.submission_time + fm_job.submission_time # be careful
            fm_job.staying_time = fm_job.pending_time + self.staying_time 
            fm_job.running_time = self.running_time 
            fm_job.pending_time = fm_job.pending_time + self.pending_time
            fm_job.num_restarts = self.num_restarts 

        return [self.modelA, self.modelB]
    

    @property
    def reweight(self, ): 
        return self.srtf_progress / (self.max_progress * 2)
    
    
    @property
    def job_number(self, ): 
        return 1 
    
    @property
    def base_weight_scale(self, ):
        return 0.5