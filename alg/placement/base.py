import os, sys
import math
import random
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..')
from server import meta
from utils import util
from alg.utils.topology import Topology
from abc import ABCMeta, abstractmethod


class AllocationPolicy(object):
    def __init__(self, policy_list):
        self.policy_list = list()
    
    def add(self, node, gpu_num):
        switch_id = node.belong_switch.id
        node_id = node.id
        filled = 1 if gpu_num + node.check_free_gpus() == self.cluster_manager.num_gpu_p_node else 0
        self.policy_list.append((switch_id, node.id, filled, gpu_num))
    
    def encapsulate(self, ):
        for policy in self.policy_list: # reorganize
            pass


    def equal(self, allocation_policy):
        assert isinstance(allocation_policy, AllocationPolicy)
        if allocation_policy.hash_val != self.hash_val:
            return False
        

class BasePlaceMent(metaclass=ABCMeta):
    __alias__ = 'base'
    def __init__(self, cluster_manager, name, **kwargs):
        self.name = name
        self.cluster_manager = cluster_manager
        self.num_gpu = cluster_manager.num_gpu
        self.num_node = cluster_manager.num_node
        self.num_switch = cluster_manager.num_switch
        self.num_gpu_p_node = cluster_manager.num_gpu_p_node
        self.num_node_p_switch = cluster_manager.num_node_p_switch
    
        
    def get_node_with_gid(self, gid):
        # s_id = int(math.floor(gid / self.num_node_p_switch))
        # n_id = int(gid % self.num_node_p_switch)
        # switch = self.cluster_manager.switch_list[s_id]
        # node = switch.node_list[n_id]
        
        found_switch, found_node, cum_node_num = None, None, 0
        for switch in self.cluster_manager.switch_list:
            if len(switch.node_list) + cum_node_num > gid:
                gid -= cum_node_num
                found_switch = switch
                found_node = switch.node_list[gid]
                break
            cum_node_num += len(switch.node_list)

        return {'switch':found_switch, 'node':found_node}


    def allocate_resource(self, job, resource, node_list, switch_list, gpu_num, cpu_num, job_num):
        node = resource['node']
        switch = resource['switch']
        switch_idx = switch.id
        if node.check_free_gpus() >= gpu_num * job_num and node.check_free_cpus() >= cpu_num * job_num:
            for _ in range(job_num):
                node_list.append(node)
                switch_list.append(switch_idx)
                if gpu_num > 0:
                    node.alloc_gpus(gpu_num, job)
                if cpu_num > 0:
                    node.alloc_cpus(cpu_num)
            return True

        return False


    def update_node_list_info(self, node, node_list, worker, ps):
        node_info = util.search_dict_list(node_list, 'node', node)
        if node_info is None:
            node_info = {
                'node' : node, 
                'worker':worker,
                'ps' : list() if worker > 0 else [ps]
            }
            node_list.append(node_info)
        else:
            if worker > 0:
                node_info['worker'] += worker
            else:
                node_info['ps'].append(ps)

    
    def fake_release_job(self, job_group):
        for job in job_group:
            if len(job['placements']) != 0:
                job['fake_placements'] = job['placements']
                job['fake_topology'] = job['topology']
                assert self.cluster_manager.try_release_resource(job) == True


    def sample_allocation_policy(self, gpu_num):
        allocate_policy_list = list()
        # intra-node, TODO shuffle
        for switch in self.cluster_manager.switch_list:
            for node in switch.node_list:
                if node.check_free_gpus() >= gpu_num:
                    allocate_policy_list.append([(node, gpu_num)])
                    break
            if len(allocate_policy_list) != 0:
                break
        
        if gpu_num < 8:
            return allocate_policy_list
        # cross-node, intra switch
        for switch in self.cluster_manager.switch_list:
            required_gpu_num = gpu_num
            # depth 2
            
        def recursive_search(cur_gpu, filled_node, cur_depth, depth, allocate_policy_set, cur_policy, switch):
            if cur_depth == depth:
                if filled_node >= depth - 1:
                    pass

                return  None

            for node in switch:
                return None
                
            self.recursive_search(gpu_num, filled_node=0, cur_depth=0, depth=2, allocate_policy_set=allocate_policy_list, cur_policy=list())

            # only allowed depth 4

        # cross-switch
        return allocate_policy_list

    def job_group_match_node_group(self, job_group):
        self.fake_release_job(job_group)
        remaining_gpu = self.cluster_manager.check_free_gpus()
        for job in job_group:
            job['gpu_score'] = 1.0  * job['gpu_num'] / remaining_gpu 
        job.sort(key=lambda x: x.__getitem__('gpu_num'))
        for job in job_group:
            allocate_policy_list= self.sample_allocation_policy(job['gpu_num'])


        # start resource allocation
        raise NotImplementedError
        self.remove_fake_holding_job(job_group)
    

    def remove_fake_holding_job(self, job_group):
        for job in job_group:
            if 'fake_placements' in job:
                del job['fake_placements']
                del job['fake_topology']
        
    def specified_placement(self, job, placements):
        raise NotImplementedError
    
    @abstractmethod
    def place_jobs(self, job):
        raise NotImplementedError
        
    
class MetaPlaceMent(object):
    __alias__ = 'meta'
    def __init__(self, placement_info, name):
        self.name = name 
        self.placement_info = placement_info
        self.cluster_keys = sorted(list(self.placement_info.keys()))
        self.node_count = 0 
        for key in self.cluster_keys: 
            cluster_instance = self.placement_info[key].cluster_manager
            self.node_count += self.node_count_of_cluster(cluster_instance=cluster_instance)

    def node_count_of_cluster(self, cluster_instance):
        node_count = 0
        for switch in cluster_instance.switch_list: 
            node_count += len(switch.node_list)
        return node_count
        
    
    def place_jobs(self, job): 
        allocation_vector = [0 for _ in range(self.node_count)]
        current_node_id = 0 
        for key in self.cluster_keys: 
            placementer = self.placement_info[key]
            placement, topology =  placementer.place_jobs(job, target_num_gpus=job.target_num_gpus, reallocate=False)
            node_count = self.node_count_of_cluster(placementer.cluster_manager)
            if sum(placement) == job.target_num_gpus: 
                allocation_vector[current_node_id:current_node_id+node_count] = placement
                job.reallocate(placement, topology=topology)
                return True 
            else: 
                current_node_id += node_count
        
        return False 
        
        
    


def PlaceMentFactory(cluster_manager, name):
    # print(BasePlaceMent.__subclasses__())
    if isinstance(cluster_manager, meta.MetaCluster): 
        placement_info = dict() 
        for key, cluster_instance in sorted(cluster_manager.cluster_instance_info.items()): 
            for subclass in BasePlaceMent.__subclasses__():
                if subclass.__alias__ == name:
                    pm = subclass(cluster_manager=cluster_instance, name=name)
                    placement_info[key] = pm 
                    break 
        return MetaPlaceMent(placement_info, name=name)
    else: 
        for subclass in BasePlaceMent.__subclasses__():
            if subclass.__alias__ == name:
                return subclass(cluster_manager=cluster_manager, name=name)

    raise NotImplementedError
