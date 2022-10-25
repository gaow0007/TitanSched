import collections
import glob
import math
import os
import pandas
import numpy as np 
from scipy.interpolate import interp1d, LinearNDInterpolator


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
        self.placements = None 
        self.scalability = None 
        self.memory_consumption = None 
        self.context_switch_overhead = None 
        # load files
        placement_file = os.path.join(trace_dir, "{}_placements.csv".format(self.name))
        if os.path.exists(placement_file): 
            self.placements = pandas.read_csv(placement_file)
            self.placements["num_nodes"] = \
                self.placements.placement.apply(lambda p: len(str(p)))
            self.placements["num_replicas"] = \
                self.placements.placement.apply(lambda p: sum(map(int, str(p))))

        scalability_file = os.path.join(trace_dir, "{}_scalability.csv".format(self.name))
        if os.path.exists(scalability_file): 
            self.scalability = pandas.read_csv(scalability_file)

        memory_consumption_file = os.path.join(trace_dir, "{}_memory.csv".format(self.name))
        if os.path.exists(memory_consumption_file): 
            self.memory_consumption = pandas.read_csv(memory_consumption_file)

        context_switch_overhead_file = os.path.join(trace_dir, "{}_context_switch.csv".format(self.name))
        if os.path.exists(context_switch_overhead_file):
            self.context_switch_overhead = pandas.read_csv(context_switch_overhead_file)

    def get_context_switch_overhead(self, placement, pipeline):
        # import pdb; pdb.set_trace() 
        # self.context_switch_overhead[placement, pipeline]
        return self.context_switch_overhead.time_cost.max() 
    
    @memoize
    def get_memory_consumption(self, local_bsz): 
        self.memory_consumption[local_bsz]
    
    @memoize
    def get_throughput(self, placement, local_bsz):
        placement = tuple(filter(None, placement))
        placement = min(placement[i:] + placement[:i]
                        for i in range(len(placement)))
        placement_id = int("".join(map(str, placement)))
        xs = ["num_nodes", "num_replicas", "local_bsz"]
        ys = ["step_time", "sync_time"]
        if placement_id in self.placements.placement.values:
            # Found in placement traces, interpolate between local_bsz.
            df = self.placements[self.placements.placement == placement_id]
            interpolator = interp1d(df.local_bsz.values, df[ys].values, axis=0)
            ret = interpolator(local_bsz)
        else:
            
            # Interpolate between num_nodes, num_replicas, and local_bsz.
            df = self.placements.groupby(xs)[xs + ys].mean()
            df = df.append(self.scalability, ignore_index=True)
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
}

class FoundationModelApplication(object): 
    def __init__(self, trace_dir, scale): 
        self.name = os.path.basename(trace_dir)
        self.scale = scale 
        self.model_name = self.name.split('@')[0]
        self.task_name = self.name.split('@')[1]
        
        # speed-related 
        self.fm_speed = FOUNDATIONMODELSPEEDS[self.model_name]
        self.max_local_bsz = self.fm_speed.placements.local_bsz.max() 
        self.min_local_bsz = self.fm_speed.placements.local_bsz.min() 
        self.metric_key = get_standarized_metric(self.task_name)
        self.progress_per_epoch = TaskScale[self.task_name]
        self.max_epochs = 10
        self.max_batch_size = 256

    
    def get_context_switch_overhead(self, placement, pipeline): 
        return self.fm_speed.get_context_switch_overhead(placement, pipeline)
    
    def get_memory_consumption(self, local_bsz): 
        self.fm_speed.memory_consumption[local_bsz]

    def get_throughput(self, placement, local_bsz):
        return self.fm_speed.get_throughput(placement, local_bsz) 
    
    def get_completion_epoch(self, batch_size, lr, target_metric):
        performance_traj = self.performance_report[batch_size][lr]
        for epoch in range(self.max_epochs): 
            metric = performance_traj[epoch][self.metric_key]
            if target_metric >= metric: 
                return epoch + 1
        return self.max_epochs
    





STATS_DIR = os.path.join(os.path.dirname(__file__), "appinfo", "fminfo")
FOUNDATIONMODELAPPLICATIONS = {
    "roberta-base@wnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@wnli"), scale="small"), 
    "roberta-base@rte": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@rte"), scale="small"), 
    "roberta-base@mrpc": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@mrpc"), scale="small"), 
    "roberta-base@stsb": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@stsb"), scale="small"), 
    "roberta-base@sst2": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@sst2"), scale="medium"), 
    "roberta-base@qnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@qnli"), scale="medium"), 
    "roberta-base@qqp": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@qqp"), scale="large"), 
    "roberta-base@mnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@mnli"), scale="large"), 
    "roberta-base@snli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-base@snli"), scale="xlarge"), 
    "roberta-large@wnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@wnli"), scale="small"), 
    "roberta-large@rte": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@rte"), scale="small"), 
    "roberta-large@mrpc": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@mrpc"), scale="small"), 
    "roberta-large@stsb": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@stsb"), scale="small"), 
    "roberta-large@sst2": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@sst2"), scale="medium"), 
    "roberta-large@qnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@qnli"), scale="medium"), 
    "roberta-large@qqp": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@qqp"), scale="large"), 
    "roberta-large@mnli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@mnli"), scale="large"), 
    "roberta-large@snli": FoundationModelApplication(os.path.join(STATS_DIR, "roberta-large@snli"), scale="xlarge"), 
}




