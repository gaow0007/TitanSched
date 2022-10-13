import os, sys
import math
import random
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from server.switch import _Switch
from server.node import _Node
from utils import util
from alg.utils.topology import Topology
from .base import BasePlaceMent


class GandivaPlaceMent(BasePlaceMent):
    __alias__ = 'gandiva'
    def __init__(self, cluster, name):
        super(GandivaPlaceMent, self).__init__(cluster=cluster, name=name, model_info=model_info)
    


    def place_jobs(self, job):
        num_gpu, node_list = job['num_gpu'], None

        assert job['num_gpu'] in self.cluster.node_g
        node_list = self.cluster.node_g[num_gpu]

        # see if exisiting resource can meet the requirements
        if len(node_list) > 0:
            node_list.sort(key=lambda x: x.__getitem__('util'))
            node_set = node_list.pop(0)
            ns_util = node_set['util']
            job_util = round(job['model']['mem_util'], 2)

            if round(ns_util + job_util, 2) <= node_set['capacity']:
                node_set['util'] = round(node_set['util'] + job_util, 2)
                node_set['jobs'].append(job)
                node_set['num_jobs'] += 1
                node_set['concurrency'] += 1
                node_list.append(node_set)
                node_list.sort(key=lambda x: x.__getitem__('util'))
                print("job [%d] starts in existing node" % job['job_id'])
                return True
            else:
                node_list.insert(0, node_set)
        
        # find new node_set
        cum_gpus = 0
        sorted_free_nodes = sorted(self.cluster.free_nodes, key=lambda node: node.check_free_gpus(), reverse=True)
        for idx, free_node in enumerate(sorted_free_nodes):
            if cum_gpus + free_node.check_free_gpus() >= num_gpu:
                node_set = {
                    'nodes': list(), 
                    'jobs' : [job], 
                    'concurrency' : 0, 
                    'num_jobs': 1, 
                    'num_gpus' : num_gpu, 
                    'capacity': int((cum_gpus + free_node.check_free_gpus()) * 1.0 / num_gpu), 
                    'util': round(job['model']['mem_util'], 2), 
                }
                # add nodes
                for i in range(idx+1):
                    free_node = sorted_free_nodes[i]
                    self.cluster.free_nodes.remove(free_node)
                    node_set['nodes'].append(free_node)
                node_list.append(node_set)
                node_list.sort(key=lambda x: x.__getitem__('util'))
                print("job [%d] starts in new node" % job['job_id'])
                return True
            cum_gpus += free_node.check_free_gpus()

        return False
        

