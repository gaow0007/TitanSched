import math 
import numpy as np 
from .template.elastic import ELASTIC_APPLICATIONS

class BatchElasticApplication(object): 
    def __init__(self, name, duration, num_gpus): 
        self.name = name 
        if self.name not in ELASTIC_APPLICATIONS.keys(): 
            self.name = 'cifar10'
        self.application_template = ELASTIC_APPLICATIONS[self.name]
        batch_size_candidates = list(self.application_template.validation.keys()) 
        selected_idx = min(int(math.log2(num_gpus)), len(batch_size_candidates) - 1)
        running_batch_size = batch_size_candidates[selected_idx] 

        template_jct = self.application_template.get_jct(num_gpus, running_batch_size)
        self.duration = duration 
        self.template_jct = template_jct
        
        self.throughput_scale = duration / template_jct 

        # copy application template 
        self.init_batch_size = self.application_template.init_batch_size
        self.max_batch_size = self.application_template.max_batch_size 
        self.min_local_bsz = self.application_template.min_local_bsz 
        self.max_local_bsz = self.application_template.max_local_bsz 
        assert self.max_batch_size >= self.min_local_bsz
        self.max_epochs = self.application_template.max_epochs
        self.target_metric = self.application_template.target_metric

        self.max_progress = self.application_template.get_progress(self.application_template.max_epochs) 


    def get_throughput(self, placement, local_bsz):
        step_time, sync_time = self.application_template.get_throughput(placement, local_bsz) 
        return step_time * self.throughput_scale, sync_time * self.throughput_scale
    
    def get_grad_stats(self, batch_size, epoch):
        return self.application_template.get_grad_stats(batch_size, epoch)
    
    def get_progress(self, epoch):
        return self.application_template.get_progress(epoch)
    
    def get_completion_epoch(self, batch_size):
        return self.application_template.get_completion_epoch(batch_size)
    
    def get_best_metric(self, batch_size, epoch):
        return self.application_template.get_best_metric(batch_size, epoch)