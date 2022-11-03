import os, sys
import math
import random
from .switch import _Switch
from .node import _Node
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..')


class MetaCluster(object): 
    def __init__(self, cluster_instance_info): 
        self.cluster_instance_info = cluster_instance_info
        self.cluster_keys = [key for key in cluster_instance_info.keys()]
        self.num_gpu_p_node = list(self.cluster_instance_info.values())[0].num_gpu_p_node
        num_node_list = list() 
        for cluster_instance in self.cluster_instance_info.values(): 
            num_node = 0 
            for switch in cluster_instance.switch_list: 
                num_node += len(switch.node_list)
            num_node_list.append(num_node)

        self.num_node = max(num_node_list)
 

    def release_gpus(self, job, status='END'):
        for placement in job['placements']:
            assert 'switch' in placement and 'nodes' in placement
            switch = self.switch_list[placement['switch']]
            assert switch.release_gpus(placement['nodes'], job) == True
        
        if status == 'END':
            job['status'] = 'END'
            print('**** job[%d] completed' % job['job_idx'])
        return True


    def release_job_resource(self, job, status='END'):
        if not hasattr(job, 'topology'): 
            import pdb; pdb.set_trace() 
            
        for placement in job.topology:
            assert 'switch' in placement and 'nodes' in placement
            found = False
            cluster_key = placement['gpu_kind']
            cluster_instance = self.cluster_instance_info[cluster_key]
            for switch in cluster_instance.switch_list:
                if switch.id == placement['switch']:
                    found = True
                    assert switch.release_job_resource(placement['nodes'], job=job) == True
                    break
            assert found == True, 'should exist in switch list'
        return True
    
    def cluster_partition(self, user_share):
        gpu_num = self.check_total_gpus()
        print(len(user_share))
        for user, share in user_share.items():
            required_gpu_num = int(share * gpu_num)
            for switch in self.switch_list:
                for node in switch.node_list:
                    if node.permission_user_list is None:
                        node.permission_user_list = [user]
                        required_gpu_num -= node.check_total_gpus()
                        if required_gpu_num <= 0: break
                if required_gpu_num <= 0: break
            assert required_gpu_num <= 0, '{} do not have resource'.format(user)

    def check_free_gpus(self, key_info=None):
        if key_info is None: 
            key_info = self.cluster_keys
        elif not isinstance(key_info, list): 
            key_info = [key_info]
        free_gpu_num = 0
        for key in key_info: 
            cluster = self.cluster_instance_info[key]
            free_gpu_num += sum([switch.check_free_gpus() for switch in cluster.switch_list])
        return free_gpu_num

    def check_free_guarante_gpus(self, user_name=None):
        return sum([switch.check_free_guarante_gpus(user_name) for switch in self.switch_list])
    
    def check_free_spot_gpus(self, user_name=None):
        return sum([switch.check_free_spot_gpus(user_name) for switch in self.switch_list])
    
    
    def check_total_gpus(self, key_info=None):
        if key_info is None: 
            key_info = self.cluster_keys
        elif not isinstance(key_info, list): 
            key_info = [key_info]
        free_gpu_num = 0
        for key in key_info: 
            cluster = self.cluster_instance_info[key]
            free_gpu_num += sum([switch.check_total_gpus() for switch in cluster.switch_list])
        return free_gpu_num

    def check_total_guarante_gpus(self, user_name=None):
        return sum([switch.check_total_guarante_gpus(user_name) for switch in self.switch_list])

    def check_total_spot_gpus(self, user_name=None):
        return sum([switch.check_total_spot_gpus(user_name) for switch in self.switch_list])

    def check_free_cpus(self, key_info=None):
        if key_info is None: 
            key_info = self.cluster_keys
        elif not isinstance(key_info, list): 
            key_info = [key_info]
        free_gpu_num = 0
        for key in key_info: 
            cluster = self.cluster_instance_info[key]
            free_gpu_num += sum([switch.check_free_cpus() for switch in cluster.switch_list])
        return free_gpu_num

    def check_total_cpus(self, key_info=None):
        if key_info is None: 
            key_info = self.cluster_keys
        elif not isinstance(key_info, list): 
            key_info = [key_info]
        free_gpu_num = 0
        for key in key_info: 
            cluster = self.cluster_instance_info[key]
            free_gpu_num += sum([switch.check_total_cpus() for switch in cluster.switch_list])
        return free_gpu_num
        