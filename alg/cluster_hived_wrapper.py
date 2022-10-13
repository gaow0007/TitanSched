import os, sys
import math
import random
sys.path.insert(0, os.path.basename(__file__) + os.sep + '..')
from utils import util
from server.cluster import _Cluster



class Cell(object):
    def __init__(self, node, resource_index_list, allow_opportunistic_user, high_prioirty_users=list(), forbidden_users=list()):
        self.node = node
        self.resource_index_list = resource_index_list
        self.resource_count = len(resource_index_list)
        self.high_prioirty_users = high_prioirty_users
        self.forbidden_users = forbidden_users
        self.allow_opportunistic_user = allow_opportunistic_user

    # user related
    def add_high_prioirty_user(self, user):
        if user not in self.high_prioirty_users:
            self.high_prioirty_users.append(user)
    
    def remove_high_priority_user(self, user):
            self.high_prioirty_users.remove(user)
    
    def add_forbidden_user(self, user):
        if user not in self.forbidden_users:
            self.forbidden_users.append(user)
    
    def remove_forbidden_user(self, user):
        self.forbidden_users.remove(user)
    
    def set_opportunistic(self, attribute):
        self.allow_opportunistic_user = attribute
    
    def empty_user_info(self, ):
        self.high_prioirty_users = list()
        self.forbidden_users = list()

    # cell related
    def isAffnitty(self, obj):
        assert isinstance(obj, Cell)
        if self.node.id != obj.node.id:
            return False

        if self.resource_count == 1:
            return min(self.resource_index_list + obj.resource_index_list) % 2 == 0

        elif self.resource_count == 2:
            return min(self.resource_index_list + obj.resource_index_list) % 4 == 0

        elif self.resource_count == 4:
            return True

        else:
            raise NotImplementedError
    

    def partition(self, ):
        buddy_cell = Cell(node=self.node, 
                          resource_index_list=self.resource_index_list[self.resource_count//2:], 
                          resource_count=self.resource_count // 2, 
                          allow_opportunistic_user=self.allow_opportunistic_user, 
                          high_prioirty_users=self.high_prioirty_users,
                          forbidden_users=self.forbidden_users)
        self.resource_index_list = self.resource_index_list[:self.resource_count // 2]
        self.resource_count //= 2
        return buddy_cell

    
    def merge(self, obj):
        assert self.isAffnitty(obj) == True, 'only affnity cell can be merged'
        self.resource_index_list += obj.resource_index_list
        self.allow_opportunistic_user = self.allow_opportunistic_user or obj.allow_opportunistic_user
        self.high_prioirty_users += obj.high_prioirty_users
        self.forbidden_users += obj.forbidden_users
        self.resource_index_list.sort()
        self.resource_count = len(self.resource_index_list)
    


class CellGroup(object):
    def __init__(self, gpu_num_index):
        self.gpu_num_index = gpu_num_index
        self.cell_list = list()
        self.upper_cell_group = None
        self.lower_cell_group = None

    @property
    def cell_count(self, ):
        return len(self.cell_list)
    
    def __getitem__(self, idx):
        return self.cell_list[idx]
    
    def __add__(self, cell_list):
        self.cell_list += cell_list

    def set_upper_cell_group(self, cell_group):
        self.upper_cell_group = cell_group
    
    def set_lower_cell_group(self, cell_group):
        self.lower_cell_group = cell_group
    
    def append(self, cell):
        assert isinstance(cell, Cell), 'only accept cell instance'
        self.cell_list.append(cell)
    
    def remove(self, cell):
        self.cell_list.remove(cell)
    
    def sort(self, key=lambda e: e.__getitem__('node').__getitem__('id')):
        self.cell_list.sort(key=key)
    
    def pop(self, ):
        return self.cell_list.pop()
    


class HivedWrapper(object):
    def __init__(self, CLUSTER, USERS):
        ''' Wrapper GPU cluster, coupled with user, partiion cluster by user requirement'''
        self.CLUSTER = CLUSTER
        self.USERS = USERS
        self.user_partition_info = dict()
        self.user_private_cluster = dict()

        # init hived cluster wrapper 
        self.init_virtual_cell()
        for user in USERS:
            self.init_user_partition(user)
    
    @property
    def num_switch(self, ):
        return self.CLUSTER.num_switch

    @property
    def num_node_p_switch(self, ):
        return self.CLUSTER.num_node_p_switch
    
    @property
    def num_gpu_p_node(self, ):
        return self.CLUSTER.num_gpu_p_node

    @property
    def num_cpu_p_node(self, ):
        return self.CLUSTER.num_cpu_p_node

    @property
    def mem_p_node(self, ):
        return self.CLUSTER.mem_p_node

    @property
    def num_node(self, ):
        return self.CLUSTER.num_node

    @property
    def num_gpu(self, ):
        return self.CLUSTER.num_gpu

    @property
    def num_cpu(self, ):
        return self.CLUSTER.num_cpu

    @property
    def free_gpu(self, ):
        return self.CLUSTER.free_gpu
    
    @property
    def mem(self, ):
        return self.CLUSTER.mem
    
    @property
    def switch_list(self, ):
        return self.CLUSTER.switch_list

    def set_spec(self, num_switch=0, num_node_p_switch=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_node=0):
        self.CLUSTER.set_spec(num_switch=num_switch, num_node_p_switch=num_node_p_switch, \
                    num_gpu_p_node=num_gpu_p_node, num_cpu_p_node=num_cpu_p_node, mem_p_node=mem_p_node)
    
    def check_free_gpus(self, ):
        return self.CLUSTER.check_free_gpus()

    def check_total_gpus(self, ):
        return self.CLUSTER.check_total_gpus()

    def check_free_cpus(self, ):
        return self.CLUSTER.check_free_cpus()
    
    def release_job_resource(self, job, status):
        return self.CLUSTER.release_job_resource(job, status)

    def init_virtual_cell(self, ):
        assert self.CLUSTER.num_gpu_p_node == 8, 'only support 8-gpu node cluster'
        for gpu_num_index in [8, 4, 2, 1]:
            setattr(self, 'cell_group_{}'.format(gpu_num_index), CellGroup(gpu_num_index))

        for a, b in zip([8, 4, 2], [4, 2, 1]):
            cell_group_a = getattr(self, 'cell_group_{}'.format(a))
            cell_group_b = getattr(self, 'cell_group_{}'.format(b))
            cell_group_a.set_lower_cell_group(cell_group_b)
            cell_group_b.set_upper_cell_group(cell_group_a)
            

        for switch in self.CLUSTER.switch_list:
            for node in switch.node_list:
                self.cell_group_8.append(Cell(node, [i for i in range(self.CLUSTER.num_gpu_p_node)], True))

    def init_user_partition(self, user, partition_info=None):
        if user in self.user_partition_info:
            self.buddy_release_by_user(self.user_partition_info[user]) # TODO release cluster
            self.user_partition_info[user] = list()
        self.add_user_partition(user, partition_info)
        
    # TODO: combine with placement strategy
    # 1. first let user to create a private cluster
    # 2. placement strategy first check private cluster is feasible and then occupy share cluster
    # 3. if other gpu occupy this resource, this resource will evict this job
    # 4. allow release resource
    # 5. resource and partition mapping mechanism

    def create_user_private_cluster(self, user):
        partition_info = self.user_partition_info[user]
        assert user not in self.user_private_cluster
        cluster_info = dict()
        cluster_info['switch_0'.format(0)] = dict()
        for i, cell in enumerate(partition_info):
            node_info = cluster_info['switch_0']
            portion = cell.resource_count * 1.0 / cell.node.check_total_gpus()
            node_info['node_{}'.format(i)] = {
                'num_gpu': cell.resource_count,
                'num_cpu': int(cell.node.check_total_cpus() * portion),
                'mem': 2 * portion, 
            }

        self.user_private_cluster[user] = _Cluster()
        self.user_private_cluster[user].set_ir_spec(cluster_info)

    def add_user_partition(self, user, partition_info=None):
        if user not in self.user_partition_info:
            self.user_partition_info[user] = list()
        
        if partition_info is not None:
            allocated = False
            if isinstance(partition_info, int):
                if self.CLUSTER.check_free_gpus() < partition_info:
                    self.user_partition_info[user] += self.buddy_allocation(partition_info)
                    for cell in self.user_partition_info[user]:
                        cell.add_high_prioirty_user(user)
                else:
                    return False

            elif isinstance(partition_info, dict):
                expected_keys = ['cell_group_{}'.format(i) for i in [8, 4, 2, 1]]
                for key in partition_info.keys():
                    assert key in expected_keys, 'provided {} not in {}'.format(key, expected_keys)
                possible_cell_count = 0
                for key in expected_keys: 
                    possible_cell_count += getattr(self, key).cell_count
                    if key in partition_info:
                        if possible_cell_count >= partition_info[key]:
                            possible_cell_count -= partition_info[key]
                        else:
                            return False
                    possible_cell_count *= 2
                for key, gpu_number_index in zip(expected_keys, [8, 4, 2, 1]):
                    if key in partition_info:
                        for _ in range(partition_info[key]):
                            self.user_partition_info[user] += self.buddy_allocation(gpu_number_index)
            else:
                raise NotImplementedError
        self.create_user_private_cluster(user)
    


    def buddy_allocation(self, required_gpu_num):
        base_gpu_num = 8
        next_gpu_num = base_gpu_num // 2
        cur_cell_group = self.cell_group_8
        if self.CLUSTER.check_free_gpus() < required_gpu_num: return False
        allocated_resource_list = list()

        
        while required_gpu_num > 0:
            
            allocated_cur_cell_resource = (required_gpu_num >= base_gpu_num) or (required_gpu_num < base_gpu_num and required_gpu_num > next_gpu_num)
            if not allocated_cur_cell_resource: 
                base_gpu_num //= 2
                next_gpu_num //= 2
                if base_gpu_num == 0: break
                continue 
            
            if cur_cell_group.cell_count == 0:
                upper_cell_group = cur_cell_group.upper_cell_group()
                while upper_cell_group is not None:
                    if upper_cell_group.cell_count > 0:
                        self.buddy_partition_by_group(upper_cell_group, cur_cell_group)
                        break
                    else:
                        upper_cell_group = upper_cell_group.upper_cell_group()

            if cur_cell_group.cell_count == 0: break

            if required_gpu_num >= base_gpu_num:
                required_gpu_num -= base_gpu_num
                allocated_resource_list.append(cur_cell_group.pop())

            elif required_gpu_num < base_gpu_num and required_gpu_num > next_gpu_num:
                sample_cell = cur_cell_group.pop()
                allocated_resource_list += self.buddy_partition_by_number(sample_cell, required_gpu_num)
                required_gpu_num = 0
                break
            
        
        assert required_gpu_num == 0, 'must meet required gpu number'
        return allocated_resource_list
    

    def buddy_partition_by_number(self, selected_cell, required_gpu_num):
        assert selected_cell.resource_count > required_gpu_num and selected_cell.resource_count // 2 < required_gpu_num

        left_cell_list = list()
        selected_cell_list = list()

        for gpu_num in [8, 4, 2, 1]:
            left_cell = None
            if selected_cell.resource_count > gpu_num:
                left_cell = selected_cell.split()
                left_cell_list.append(left_cell)
            
            if required_gpu_num >= gpu_num:
                assert gpu_num == selected_cell.resource_count
                required_gpu_num -= gpu_num
                selected_cell_list.append(selected_cell)
                selected_cell = left_cell_list.pop()
            if required_gpu_num == 0:
                break
        
        # reassign to corresponding group
        for gpu_num_index in [8, 4, 2, 1]:
            cell_group = getattr(self, 'cell_group_{}'.format(gpu_num_index))
            for cell in left_cell_list:
                if cell.resource_count == gpu_num_index:
                    cell_group.append(cell)
        
        return selected_cell_list
        

    def buddy_partition_by_group(self, cell_group, target_cell_group):
        assert cell_group.gpu_num_index > target_cell_group.gpu_num_index
        assert cell_group.resource > 0
        while cell_group.gpu_num_index != target_cell_group.gpu_num_index:
            sample_cell = cell_group.pop()
            sibling_cell = sample_cell.partition()
            cell_group = cell_group.lower_cell_group()
            cell_group += [sample_cell, sibling_cell]

    def buddy_release_by_user(self, release_cell_list):
        # return resource to cluster
        for release_cell in release_cell_list:
            release_cell.empty_user_info()
            cell_group = getattr(self, 'cell_group_{}'.format(release_cell_list.resource_count))
            cell_group.append(release_cell)
        
        cur_cell_group = self.cell_group_1
        for _ in [2, 4, 8]:
            combine_pair_list = list()
            cur_cell_group.sort()
            for cella, cellb in zip(cur_cell_group[:-1], cur_cell_group[1:]):
                if cella.isAffnitty(cellb):
                    combine_pair_list.append((cella, cellb))
            # combine
            for (cella, cellb) in combine_pair_list:
                merge_cell = cella.merge(cellb)
                cur_cell_group.remove(cella)
                cur_cell_group.remove(cellb)
                cur_cell_group.upper_cell_group().append(merge_cell)
            cur_cell_group = cur_cell_group.upper_cell_group()
        
        # TODO: post check

