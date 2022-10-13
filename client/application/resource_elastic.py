import numpy as np 



class ResourceElasticApplication(object): 
    def __init__(self, name, duration, num_gpus):
        self.name = name
        self.max_progress = duration
        self.num_gpus = num_gpus 
        self.power = np.random.rand() * 0.1 + 0.9
        self.max_num_gpus = num_gpus * 4 
        self.min_num_gpus = 1 # max(1, num_gpus // 4)
    
    def get_throughput(self, placement): 
        if sum(placement) >= self.num_gpus: 
            return (sum(placement) * 1.0 / self.num_gpus) ** self.power 
        else: 
            return sum(placement) * 1.0 / self.num_gpus 

