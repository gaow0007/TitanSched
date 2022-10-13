import os, sys
import math
import random
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from server.switch import _Switch
from server.node import _Node
from utils import util
from alg.utils.topology import Topology
from .base import BasePlaceMent


class ConsolidateRandomPlaceMent(BasePlaceMent):
    __alias__ = 'consolidate_random'
    def __init__(self, cluster_manager, name):
        super(ConsolidateRandomPlaceMent, self).__init__(cluster_manager=cluster_manager, name=name)
    
    
    def select_fewest_node_list(self, target_num_gpus):
        best_select_list = list()
        
        # select in a single switch
        for switch in self.cluster_manager.switch_list:
            free_gpu_num = switch.check_free_gpus()
            reverse = False if switch.node_list[0].check_total_gpus() >= target_num_gpus else True
            if free_gpu_num >= target_num_gpus:
                # method 1
                if reverse == True:
                    node_list = sorted(switch.node_list, key=lambda e: e.check_free_gpus(), reverse=reverse)
                else:
                    nonzero = lambda x: x if x >= target_num_gpus else 1000
                    node_list = sorted(switch.node_list, key=lambda e: nonzero(e.check_free_gpus()), reverse=reverse)
                select_list = list()
                
                local_target_num_gpus = target_num_gpus
                for node in node_list:
                    free_gpu_num = node.check_free_gpus()
                    if free_gpu_num == 0: continue
                    if free_gpu_num < local_target_num_gpus:
                        node_info = (node, free_gpu_num)
                    else:
                        node_info = (node, local_target_num_gpus)

                    local_target_num_gpus -= node_info[1]
                    select_list.append(node_info)
                    if local_target_num_gpus == 0:
                        break
                
                if local_target_num_gpus == 0 and (len(best_select_list) == 0 or (len(best_select_list) > len(select_list))):
                    best_select_list = select_list
                # print('length {}, required_gpu {}'.format(len(best_select_list), target_num_gpus))

        if len(best_select_list) == 0:
            local_target_num_gpus = target_num_gpus
            reverse = False if self.cluster_manager.switch.node_list[0].check_total_gpus() >= target_num_gpus else True
            switch_list = sorted(self.cluster_manager.switch_list, key=lambda e: e.check_free_gpus(), reverse=reverse)
            
            for switch in switch_list:
                free_gpu_num = switch.check_free_gpus()
                if free_gpu_num >= local_target_num_gpus:
                    node_list = sorted(switch.node_list, key=lambda e: e.check_free_gpus(), reverse=reverse)
                else:
                    node_list = switch.node_list

                for node in node_list:
                    free_gpu_num = node.check_free_gpus()
                    if free_gpu_num == 0:
                        continue
                    if free_gpu_num < local_target_num_gpus:
                        node_info = (node, free_gpu_num)
                    else:
                        node_info = (node, local_target_num_gpus)

                    local_target_num_gpus -= node_info[1]
                    best_select_list.append(node_info)
                    if local_target_num_gpus == 0:
                        break
                if local_target_num_gpus == 0:
                    break
        return best_select_list
  

    def place_jobs(self, job):
        '''
        consolidate first, but randomly pick machines;
        if cross machines, still try to consolidate.
        if can't consolidate, consider spreed the jobs;
        also PS is randomly placed on the selected machines
        '''
        # early return false
        if self.cluster_manager.check_free_gpus() < job.target_num_gpus: return False

        # place as few nodes as possible
        w_node_list, w_switch_list = list(), list()
        demand_node_list = self.select_fewest_node_list(job.target_num_gpus)

        placement = list() 
        for switch_intance in self.cluster_manager.switch_list: 
            for _ in range(len(switch_intance.node_list)): 
                placement.append(0)
        
        for node_info in demand_node_list:
            node, need_gpu = node_info
            switch = node.belong_switch
            idx = switch.id * self.cluster_manager.num_node_p_switch +  node.id
            placement[idx] += need_gpu
            allocated = self.allocate_resource(job=job, resource={'node':node, 'switch':switch}, node_list=w_node_list, \
                switch_list=w_switch_list, gpu_num=1, cpu_num=2, job_num=need_gpu)
            assert allocated == True, 'should exist enough gpu resource'

        
        assert len(w_node_list) == job.target_num_gpus
        topology = list() 
        for i, (s_id, node) in enumerate(zip(w_switch_list, w_node_list)):

            node_dict = {
                'id' : node.id, 
                'node_instance' : node, 
                'num_gpu' : 1,
                'num_cpu' : 2,
                'mem' : 0 if not hasattr(job.application,'mem_util') else job.application.mem_util,
                'tasks': list(), 
            }
            topology.append({
                'switch' : s_id, 
                'nodes' : [node_dict],
            })
        
        job.reallocate(placement, topology=topology) # , topology=Topology(job=job, placements=topology))
        return True

