import numpy as np 


class FoundationModelApplication(object): 
    def __init__(self, name, data_scale, iteration): 
        self.name = name 
        self.data_scale = data_scale
        self.max_progress = iteration 
        self.placement2context_switch_overhead_no_pipeline = {
            1 : 18,
            2: 22,  
            4: 29, 
            8: 46, 
        }
        self.placement2context_switch_overhead_with_pipeline = {
            1: 5,
            2: 7,  
            4: 5, 
            8: 6, 
        }
        
    def get_context_switch_overhead(self, placement, pipeline): 
        if not pipeline: 
            return self.placement2context_switch_overhead_no_pipeline[sum(placement)]
        else: 
            return self.placement2context_switch_overhead_with_pipeline[sum(placement)]

    def get_throughput(self, placement, local_bsz):
        if sum(placement) == 1: 
            return 1 / 30.883228171955455 # iter / seconds 
        elif sum(placement) == 2: 
            return 1 / 16.883228171955455
        elif sum(placement) == 4: 
            return 1 / 9.52039733800021
        elif sum(placement) == 8: 
            return 1 / 5.52039733800021
        else: 
            raise NotImplementedError 
            
    def get_max_throughput(self, placement):
        if sum(placement) == 1: 
            return 1 / 30.883228171955455
        elif sum(placement) == 2: 
            return 1 / 16.883228171955455
        elif sum(placement) == 4: 
            return 1 / 9.52039733800021
        elif sum(placement) == 8: 
            return 1 / 5.52039733800021