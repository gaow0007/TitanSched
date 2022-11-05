import collections
import glob
import math
import os
import pandas
import numpy as np 
from scipy.interpolate import interp1d, LinearNDInterpolator

DEFAULT_GPU_KIND="V100"

def get_standarized_metric(task_name): 
    return task_name

def memoize(f):
    memo = {}
    def helper(*x):
        if x not in memo:
            memo[x] = f(*x)
        return memo[x]
    return helper

class FoundationModelSpeed(object): 
    def __init__(self, trace_dir, name):
        self.name = name 
        self.placements = {} 
        self.scalability = {} 
        self.memory_consumption = {} 
        self.context_switch_overhead = {} 
        # load files
        for GPU_KIND in ['V100', 'A100']: 
            placement_file = os.path.join(trace_dir, GPU_KIND, "{}_placements.csv".format(self.name))
            if os.path.exists(placement_file): 
                placements = pandas.read_csv(placement_file)
                placements["num_nodes"] = \
                    placements.placement.apply(lambda p: len(str(p)))
                placements["num_replicas"] = \
                    placements.placement.apply(lambda p: sum(map(int, str(p))))
                self.placements[GPU_KIND] = placements

            scalability_file = os.path.join(trace_dir, GPU_KIND, "{}_scalability.csv".format(self.name))
            if os.path.exists(scalability_file): 
                scalability = pandas.read_csv(scalability_file)
                self.scalability[GPU_KIND] = scalability

            memory_consumption_file = os.path.join(trace_dir, "{}_memory.csv".format(self.name))
            if os.path.exists(memory_consumption_file): 
                memory_consumption = pandas.read_csv(memory_consumption_file)
                self.memory_consumption[GPU_KIND] = memory_consumption

            context_switch_overhead_file = os.path.join(trace_dir, GPU_KIND, "{}_context_switch.csv".format(self.name))
            if os.path.exists(context_switch_overhead_file):
                context_switch_overhead = pandas.read_csv(context_switch_overhead_file)
                self.context_switch_overhead[GPU_KIND] = context_switch_overhead

    def get_context_switch_overhead(self, GPU_KIND, placement, pipeline):
        # import pdb; pdb.set_trace() 
        # self.context_switch_overhead[placement, pipeline]
        return float(self.context_switch_overhead[GPU_KIND].time_cost.max())
    
    @memoize
    def get_memory_consumption(self, GPU_KIND, local_bsz): 
        self.memory_consumption[GPU_KIND][local_bsz]
    
    @memoize
    def get_throughput(self, GPU_KIND, placement, local_bsz):
        # print(placement, local_bsz)
        placement = tuple(filter(None, placement))
        placement = min(placement[i:] + placement[:i]
                        for i in range(len(placement)))
        placement_id = int("".join(map(str, placement)))
        xs = ["num_nodes", "num_replicas", "local_bsz"]
        ys = ["step_time", "sync_time"]
        if placement_id in self.placements[GPU_KIND].placement.values:
            # Found in placement traces, interpolate between local_bsz.
            df = self.placements[GPU_KIND][self.placements[GPU_KIND].placement == placement_id]
            interpolator = interp1d(df.local_bsz.values, df[ys].values, axis=0)
            ret = interpolator(local_bsz)
        else:
            # Interpolate between num_nodes, num_replicas, and local_bsz.
            df = self.placements[GPU_KIND].groupby(xs)[xs + ys].mean()
            df = df.append(self.scalability[GPU_KIND], ignore_index=True)
            num_nodes, num_replicas = len(placement), sum(placement)
            num_nodes = min(num_nodes, 16)
            
            interpolator = LinearNDInterpolator(df[xs].values, df[ys].values)
            ret = interpolator([num_nodes, num_replicas, local_bsz])[0]
        assert sum(ret) == sum(ret), "{} {} {}".format(self.name, placement, local_bsz)
        return ret

SPEED_DIR = os.path.join(os.path.dirname(__file__), "appinfo", "fminfo")
FOUNDATIONMODELSPEEDS = {
    "vit" : FoundationModelSpeed(SPEED_DIR, "vit"), 
    "vit-large": FoundationModelSpeed(SPEED_DIR, "vit-large"), 
    "roberta-base": FoundationModelSpeed(SPEED_DIR, "roberta-base"), 
    "roberta-large": FoundationModelSpeed(SPEED_DIR, "roberta-large"), 
}

# //////////////////////////////////////////////////////////////////////////////////////////////////////
from common.prefix_utils import generate_prefix
from common.applications import (
    HPOTaskingInfo, 
    MultiTaskingInfo, 
    TransferTaskingInfo, 
    metrick_key_generator,
    query_index)

class FoundationModelStats(object): 
    def __init__(self, ): 
        self.hpo_info = HPOTaskingInfo(load_cache=True)
        self.mtask_info = MultiTaskingInfo(load_cache=True) 
        self.transfer_info = TransferTaskingInfo(load_cache=True) 
    
    def query_epoch(self, model, dataset, lr, gradient_steps, metric_key, target_metric): 
        return self.hpo_info.query_epoch(model, dataset, lr, gradient_steps, metric_key, target_metric)


FMStats = FoundationModelStats() 
# //////////////////////////////////////////////////////////////////////////////////////////////////////
TaskScale = {
    'wnli' : 159,
    'rte' : 623,
    'mrpc' : 917,
    'stsb' : 1438,
    'sst2' : 16838,
    'qnli' : 26186,
    'qqp' : 90962,
    'mnli' : 98176,
    'snli' : 137344,
    'ag_news' : 30000,
    'wikitext-103':59061,
    # for image classification
    "Bingsu#Cat_and_Dog": 8000,
    "frgfm#imagenette": 9469,
    "Bingsu#Human_Action_Recognition": 12600,
    "cifar100": 50000,
    "fashion_mnist": 60000,
    "mnist": 60000,
    "food101": 75750,
    "imagenet-split-0": 25000, 
    "imagenet-split-1": 75000, 
    "imagenet-split-2": 100000, 
}

class FoundationModelApplication(object): 
    def __init__(self, trace_dir, scale): 
        self.name = os.path.basename(trace_dir)
        self.scale = scale 
        self.model_name = self.name.split('@')[0]
        self.task_name = self.name.split('@')[1]
        
        # speed-related 
        self.fm_speed = FOUNDATIONMODELSPEEDS[self.model_name]

        # stats-related 
        
        self.max_local_bsz = int(self.fm_speed.placements[DEFAULT_GPU_KIND].local_bsz.max())
        self.min_local_bsz = int(self.fm_speed.placements[DEFAULT_GPU_KIND].local_bsz.min())
        self.metric_key = FMStats.hpo_info.standarized_metric(self.task_name)
        self.progress_per_epoch = TaskScale[self.task_name]
        self.max_epochs = 10
        self.max_batch_size = 1024

    
    def get_context_switch_overhead(self, GPU_KIND, placement, pipeline): 
        return self.fm_speed.get_context_switch_overhead(GPU_KIND, placement, pipeline)
    
    def get_memory_consumption(self, local_bsz): 
        self.fm_speed.memory_consumption[local_bsz]

    def get_throughput(self, GPU_KIND, placement, local_bsz):
        return self.fm_speed.get_throughput(GPU_KIND, placement, local_bsz) 
    
    def get_completion_epoch(self,  **kwargs):
        target_metric = kwargs.get('target_metric')
        assert target_metric is not None 
        if kwargs.get('transfer') == True: 
            insert_key = FMStats.transfer_info.generate_dataset_key(kwargs.get('taskA'), kwargs.get('taskB'))
            if insert_key not in FMStats.transfer_info.performance_report[self.model_name]: 
                # import pdb; pdb.set_trace() 
                return 1000
            performance_traj = FMStats.transfer_info.performance_report[self.model_name][insert_key] 
            metric_info = performance_traj[self.metric_key]
            epoch_info = performance_traj[self.metric_key + '_epoch']
        elif kwargs.get('mtask') == True: 
            insert_key = FMStats.mtask_info.generate_dataset_key(kwargs.get('taskA'), kwargs.get('taskB'))
            performance_traj = FMStats.mtask_info.performance_report[self.model_name][insert_key] 
            # print(performance_traj.keys())
            metric_key = '{}_{}'.format(self.task_name, self.metric_key)
            metric_info = performance_traj[metric_key]
            epoch_info = performance_traj[metric_key + '_epoch']
        else: 
            lr = kwargs.get('lr')
            gradient_steps = kwargs.get('gradient_steps')
            performance_traj =  FMStats.hpo_info.performance_report[self.model_name][self.task_name][lr][gradient_steps]
            metric_info = performance_traj[self.metric_key]
            epoch_info = performance_traj[self.metric_key + '_epoch']


        min_epoch = None 
        for epoch, metric in zip(epoch_info, metric_info): 
            if target_metric <= metric: 
                if min_epoch is None: 
                    min_epoch = epoch 
                else: 
                    min_epoch = min(epoch, min_epoch)
        
        return self.max_epochs if min_epoch is None else min_epoch
    
    def query_epoch(self, target_lr=None, target_gradient_steps=None, target_epoch=None):
        assert target_lr is not None and target_gradient_steps is not None
        performance_traj =  FMStats.hpo_info.performance_report[self.model_name][self.task_name][target_lr][target_gradient_steps]
        metric_info = performance_traj[self.metric_key]
        epoch_info = [int(round(epoch)) for epoch in performance_traj[self.metric_key + '_epoch']]
        epoch_metric_list = sorted([(epoch, metric) for (epoch, metric) in zip(epoch_info, metric_info)])
        max_metric = 0
        for epoch, metric in epoch_metric_list: 
            max_metric = max(metric, max_metric)
            if epoch >= target_epoch: 
                return max_metric
        import pdb; pdb.set_trace() 


    def query_index(self, ): 
        return query_index(self.task_name)



STATS_DIR = os.path.join(os.path.dirname(__file__), "appinfo", "fminfo", "V100")
FOUNDATIONMODELAPPLICATIONS = {
    # ////////////////////////// -- language models 
    "roberta-base@wnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@wnli"), scale="small"), 
    "roberta-base@rte": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@rte"), scale="small"), 
    "roberta-base@mrpc": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@mrpc"), scale="small"), 
    "roberta-base@stsb": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@stsb"), scale="small"), 
    "roberta-base@sst2": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@sst2"), scale="medium"), 
    "roberta-base@qnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@qnli"), scale="medium"), 
    "roberta-base@qqp": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@qqp"), scale="large"), 
    "roberta-base@mnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@mnli"), scale="large"), 
    "roberta-base@snli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@snli"), scale="large"), 
    "roberta-large@wnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@wnli"), scale="small"), 
    "roberta-large@rte": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@rte"), scale="small"), 
    "roberta-large@mrpc": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@mrpc"), scale="small"), 
    "roberta-large@stsb": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@stsb"), scale="small"), 
    "roberta-large@sst2": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@sst2"), scale="medium"), 
    "roberta-large@qnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@qnli"), scale="medium"), 
    "roberta-large@qqp": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@qqp"), scale="large"), 
    "roberta-large@mnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@mnli"), scale="large"), 
    "roberta-large@snli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@snli"), scale="large"), 
    # ////////////////////////// -- vision models 
    "vit@Bingsu#Cat_and_Dog": FoundationModelApplication(os.path.join(STATS_DIR, "vit@Bingsu#Cat_and_Dog"), scale="small"), 
    "vit@frgfm#imagenette": FoundationModelApplication(os.path.join(STATS_DIR, "vit@frgfm#imagenette"), scale="small"), 
    "vit@cifar100": FoundationModelApplication(os.path.join(STATS_DIR, "vit@cifar100"), scale="medium"), 
    "vit@fashion_mnist": FoundationModelApplication(os.path.join(STATS_DIR, "vit@fashion_mnist"), scale="medium"), 
    "vit@mnist": FoundationModelApplication(os.path.join(STATS_DIR, "vit@mnist"), scale="medium"), 
    "vit@food101": FoundationModelApplication(os.path.join(STATS_DIR, "vit@food101"), scale="large"), 
    "vit@imagenet-split-0": FoundationModelApplication(os.path.join(STATS_DIR, "vit@imagenet-split-0"), scale="medium"), 
    "vit@imagenet-split-1": FoundationModelApplication(os.path.join(STATS_DIR, "vit@imagenet-split-1"), scale="large"), 
    "vit@imagenet-split-2": FoundationModelApplication(os.path.join(STATS_DIR, "vit@imagenet-split-2"), scale="large"), 
    "vit-large@Bingsu#Cat_and_Dog": FoundationModelApplication(os.path.join(STATS_DIR, "vit-large@Bingsu#Cat_and_Dog"), scale="small"), 
    "vit-large@frgfm#imagenette": FoundationModelApplication(os.path.join(STATS_DIR, "vit-large@frgfm#imagenette"), scale="small"), 
    "vit-large@cifar100": FoundationModelApplication(os.path.join(STATS_DIR, "vit-large@cifar100"), scale="medium"), 
    "vit-large@fashion_mnist": FoundationModelApplication(os.path.join(STATS_DIR, "vit-large@fashion_mnist"), scale="medium"), 
    "vit-large@mnist": FoundationModelApplication(os.path.join(STATS_DIR, "vit-large@mnist"), scale="medium"), 
    "vit-large@food101": FoundationModelApplication(os.path.join(STATS_DIR, "vit-large@food101"), scale="large"), 
    "vit-large@imagenet-split-0": FoundationModelApplication(os.path.join(STATS_DIR, "vit-large@imagenet-split-0"), scale="medium"), 
    "vit-large@imagenet-split-1": FoundationModelApplication(os.path.join(STATS_DIR, "vit-large@imagenet-split-1"), scale="medium"), 
    "vit-large@imagenet-split-2": FoundationModelApplication(os.path.join(STATS_DIR, "vit-large@imagenet-split-2"), scale="large"), 
}




