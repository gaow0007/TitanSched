from heapq import merge
from mip import *
import numpy as np
import copy
import math
from .titan_utils import allocation2num


class TitanMultiTaskAdaptivitySolver(object):
    def __init__(self, method, logger):
        self.method = method
        self.logger = logger

    def job_selection(self, required_resource_list, weight_per_allocation_list, equalivent_allocation_list, unique_job_num, \
                            mtask_required_resource_list, mtask_weight_per_allication_list, mtask_equalivent_allocation_list, mtask_unique_job_num, \
                            transfer_required_resource_list, transfer_weight_per_allication_list, transfer_equalivent_allocation_list, transfer_unique_job_num, \
                            temporal_transfer_required_resource_list, temporal_transfer_weight_per_allication_list, temporal_transfer_equalivent_allocation_list, temporal_transfer_unique_job_num, \
                            cluster_capacity, max_seconds=1, power=1): 

        m = Model(solver_name = GRB)
        var_len = len(required_resource_list) + len(mtask_required_resource_list) + len(transfer_required_resource_list) + len(temporal_transfer_required_resource_list)
        X = [m.add_var(var_type=BINARY) for i in range(var_len)]
        tot_weight_list = weight_per_allocation_list + mtask_weight_per_allication_list + transfer_weight_per_allication_list + temporal_transfer_weight_per_allication_list
        if len(tot_weight_list) != var_len: 
            import pdb; pdb.set_trace() 
        # obj_list = [X[i] * tot_weight_list[i] for i in range(var_len)]
        obj_list = [X[i] * (tot_weight_list[i] ** power) for i in range(var_len)]

        if power > 0: 
            m.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list)))) 
        else: 
            m.objective = minimize(xsum(obj_list[i] for i in range(len(obj_list)))) 



        # m.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list)))) 
        tot_resource_list = required_resource_list + mtask_required_resource_list + transfer_required_resource_list + temporal_transfer_required_resource_list
        for key in cluster_capacity.keys(): 
            resource_list = [sum(tot_resource_list[i][key]) if key in tot_resource_list[i] else 0 for i in range(var_len)]
            m += xsum(X[i] * resource_list[i] for i in range(var_len)) <= cluster_capacity[key] 
        
        # m += xsum(X[i] * tot_resource_list[i] for i in range(var_len)) <= cluster_capacity

        # single-task intra resource allocation constraint
        left, right = 0, 0
        left_list, right_list = list(), list() 
        for i in range(unique_job_num): 
            reference_allocation = equalivent_allocation_list[left]
            for j in range(left, len(equalivent_allocation_list)): 
                if equalivent_allocation_list[j] != reference_allocation: 
                    right = j 
                    break 
            if equalivent_allocation_list[j] == reference_allocation: right = j + 1 
            m += xsum(X[j] for j in range(left, right)) == 1
            left_list.append(left) 
            right_list.append(right)
            left = right 

        # multi-task intra resource allocation constraint
        left, right = 0, 0 
        merge_left_list, merge_right_list = list(), list() 
        for i in range(mtask_unique_job_num): 
            reference_allocation = mtask_equalivent_allocation_list[left]
            for j in range(left, len(mtask_equalivent_allocation_list)): 
                if mtask_equalivent_allocation_list[j] != reference_allocation: 
                    # self.logger.info('not equal between {} and {}'.format(mtask_equalivent_allocation_list[j], reference_allocation))
                    right = j
                    break 
            
            if mtask_equalivent_allocation_list[j] == reference_allocation: right = j + 1 
            # print('j == {}'.format(j))
            # print('*' * 30)
            # print(reference_allocation, mtask_equalivent_allocation_list[right], len(mtask_equalivent_allocation_list), j)
            # # print(mtask_equalivent_allocation_list[right] == reference_allocation)
            self.logger.info('mtask left {}, right {}, reference_allocation {}'.format(left, right, reference_allocation))
            m += xsum(X[j + len(required_resource_list)] for j in range(left, right)) == 1 # be careful about index shift 
            # print('constraint for mtask {}'.format([j + len(required_resource_list) for j in range(left, right)]))
            merge_left_list.append(left)
            merge_right_list.append(right)
            left = right 

        # transfer job intra job constraint 
        start_index = len(required_resource_list) + len(mtask_required_resource_list)
        transfer_left_list, transfer_right_list = list(), list() 
        left, right = 0, 0
        for i in range(transfer_unique_job_num): 
            if left >= len(transfer_equalivent_allocation_list): 
                import pdb; pdb.set_trace() 
            reference_allocation = transfer_equalivent_allocation_list[left]
            for j in range(left, len(transfer_equalivent_allocation_list)): 
                if transfer_equalivent_allocation_list[j] != reference_allocation: 
                    right = j 
                    break 
            if transfer_equalivent_allocation_list[j] == reference_allocation: right = j + 1 
            m += xsum(X[j+start_index] for j in range(left, right)) == 1
            transfer_left_list.append(left) 
            transfer_right_list.append(right)
            left = right 

        # temporal transfer job intra job constraint
        start_index = len(required_resource_list) + len(mtask_required_resource_list) + len(transfer_required_resource_list)
        temporal_transfer_left_list, temporal_transfer_right_list = list(), list() 
        left, right = 0, 0

        # str_temporal_transfer_equalivent_allocation_list = ['-'.join([str(alloc) for alloc in eq_alloc]) for eq_alloc in temporal_transfer_equalivent_allocation_list]
        # if np.unique(str_temporal_transfer_equalivent_allocation_list) == temporal_transfer_unique_job_num: 
        #     import pdb; pdb.set_trace() 
            
        for i in range(temporal_transfer_unique_job_num): 
            # print('left {}, right {}'.format(left, right))
            if left >= len(temporal_transfer_equalivent_allocation_list): 
                import pdb; pdb.set_trace() 
            reference_allocation = temporal_transfer_equalivent_allocation_list[left]
            for j in range(left, len(temporal_transfer_equalivent_allocation_list)): 
                if temporal_transfer_equalivent_allocation_list[j] != reference_allocation: 
                    right = j 
                    break 
            if temporal_transfer_equalivent_allocation_list[j] == reference_allocation: right = j + 1 
            m += xsum(X[j+start_index] for j in range(left, right)) == 1
            temporal_transfer_left_list.append(left) 
            temporal_transfer_right_list.append(right)
            left = right 

        # single-task and mult-task resource allocation constraint
        for i in range(unique_job_num): 
            reference_single_allocation = equalivent_allocation_list[left_list[i]]
            conflict_list = list() 
            for single_i in range(len(required_resource_list)): 
                if equalivent_allocation_list[single_i] == reference_single_allocation and allocation2num(required_resource_list[single_i]) != 0: 
                    conflict_list.append(single_i)
            
            single_length = len(required_resource_list)
            for mtask_i in range(len(mtask_required_resource_list)): 
                if reference_single_allocation in mtask_equalivent_allocation_list[mtask_i] and allocation2num(mtask_required_resource_list[mtask_i]) != 0: 
                    conflict_list.append(mtask_i + single_length)
            
            mtask_length = len(mtask_required_resource_list)
            for transfer_i in range(len(transfer_required_resource_list)): 
                if reference_single_allocation in transfer_equalivent_allocation_list[transfer_i] and allocation2num(transfer_required_resource_list[transfer_i]) != 0: 
                    conflict_list.append(transfer_i + mtask_length + single_length)
            

            transfer_length = len(transfer_required_resource_list)
            for temporal_transfer_i in range(len(temporal_transfer_equalivent_allocation_list)): 
                if reference_single_allocation in temporal_transfer_equalivent_allocation_list[temporal_transfer_i] and \
                    allocation2num(temporal_transfer_required_resource_list[temporal_transfer_i]) != 0: 
                    conflict_list.append(temporal_transfer_i + mtask_length + single_length + transfer_length)
            m += xsum(X[j] for j in conflict_list) <= 1

        # # single-task and mult-task resource allocation constraint
        # for i in range(mtask_unique_job_num): 
        #     merge_idx_list = mtask_equalivent_allocation_list[merge_left_list[i]]
            
        #     for merge_idx in merge_idx_list: 
        #         conflict_list = list() 
        #         left, right = left_list[merge_idx], right_list[merge_idx]
        #         # import pdb; pdb.set_trace() 
        #         for j in range(left, right): 
        #             if required_resource_list[j] != 0: 
        #                 conflict_list.append(j)
            
        #         left, right = merge_left_list[i], merge_right_list[i]
        #         for j in range(left, right): 
        #             if mtask_required_resource_list[j] != 0:
        #                 conflict_list.append(len(required_resource_list) + j) 

        #         print('*' * 30)
        #         print('conflict_list {}'.format(conflict_list))
        #         m += xsum(X[j] for j in conflict_list) <= 1


        m.optimize(max_seconds=max_seconds)
        allocated_gpu_solution = list() 
        left, right = 0, 0
        for i in range(unique_job_num):
            allocated = False 
            
            for j in range(left_list[i], right_list[i]): 
                res = X[j].x
                if res is not None and res > 0.5: 
                    allocated_gpu_solution.append(required_resource_list[j])
                    allocated = True 
            if not allocated: 
                allocated_gpu_solution.append(0) 
        
        multi_task = False 
        for i in range(mtask_unique_job_num): 
            allocated = False 
            for j in range(merge_left_list[i], merge_right_list[i]): 
                res = X[j + len(required_resource_list)].x 
                if res is not None and res > 0.5: 
                    multi_task = True 
                    allocated_gpu_solution.append(mtask_required_resource_list[j])
                    allocated = True 
            if not allocated: 
                allocated_gpu_solution.append(None) 
        
        for i in range(transfer_unique_job_num): 
            allocated = False 
            for j in range(transfer_left_list[i], transfer_right_list[i]): 
                res = X[j + len(required_resource_list) + len(mtask_required_resource_list)].x 
                if res is not None and res > 0.5: 
                    allocated_gpu_solution.append(transfer_required_resource_list[j])
                    allocated = True 
            if not allocated: 
                allocated_gpu_solution.append(None)

        for i in range(temporal_transfer_unique_job_num): 
            allocated = False 
            for j in range(temporal_transfer_left_list[i], temporal_transfer_right_list[i]): 
                res =  X[j + len(required_resource_list) + len(mtask_required_resource_list) + len(transfer_required_resource_list)].x 
                if res is not None and res > 0.5: 
                    allocated_gpu_solution.append(temporal_transfer_required_resource_list[j])
                    allocated = True 
            
            if not allocated: 
                allocated_gpu_solution.append(None)
        
        # tot_resource_list = required_resource_list + mtask_required_resource_list + transfer_required_resource_list
        # tot_allocation_list = equalivent_allocation_list + mtask_equalivent_allocation_list + transfer_equalivent_allocation_list
        
        # for k in range(var_len): 
        #     ident = 'mtask' if k >= len(required_resource_list) else 'single'
        #     if ident == 'mtask': 
        #         ident = 'transfer_task' if k >= len(required_resource_list) + len(mtask_required_resource_list) else 'mtask'
        #     if isinstance(tot_allocation_list[k], list) and 10 in tot_allocation_list[k]: 
        #         self.logger.info('k, {}, x {}, resource {}, ident {}, weight {}, eq {}'.format(k, X[k].x, tot_resource_list[k], ident, tot_weight_list[k], tot_allocation_list[k]))
        #     elif isinstance(tot_allocation_list[k], int) and tot_allocation_list[k] == 10: 
        #         self.logger.info('k, {}, x {}, resource {}, ident {}, weight {}, eq {}'.format(k, X[k].x, tot_resource_list[k], ident, tot_weight_list[k], tot_allocation_list[k]))

        # print([X[k].x for k in range(var_len)])
        # print(required_resource_list + mtask_required_resource_list)

        # if multi_task: 
        #     self.logger.info("allocated_gpu_solution is {}".format(allocated_gpu_solution))
        #     import pdb; pdb.set_trace() 
        # self.logger.info("allocated_gpu_solution is {}".format(allocated_gpu_solution))
        # self.logger.info("total gpu {}".format(sum(allocated_gpu_solution)))
        # if sum(allocated_gpu_solution) > 0 and var_len > 30: 
        #     import pdb; pdb.set_trace() 

        return allocated_gpu_solution

# k, 0, x 1.0, resource 0, ident single
# k, 1, x 0.0, resource 1, ident single
# k, 2, x 0.0, resource 2, ident single
# k, 3, x 0.0, resource 3, ident single
# k, 4, x 0.0, resource 4, ident single
# k, 5, x 0.0, resource 8, ident single
# k, 6, x 0.0, resource 12, ident single
# k, 7, x 0.0, resource 16, ident single
# k, 8, x 0.0, resource 20, ident single
# k, 9, x 0.0, resource 24, ident single
# k, 10, x 0.0, resource 28, ident single
# k, 11, x 0.0, resource 32, ident single
# k, 12, x 1.0, resource 0, ident single
# k, 13, x 0.0, resource 1, ident single
# k, 14, x 0.0, resource 2, ident single
# k, 15, x 0.0, resource 3, ident single
# k, 16, x 0.0, resource 4, ident single
# k, 17, x 0.0, resource 8, ident single
# k, 18, x 1.0, resource 0, ident single
# k, 19, x 0.0, resource 1, ident mtask
# k, 20, x 0.0, resource 2, ident mtask
# k, 21, x 0.0, resource 3, ident mtask
# k, 22, x 0.0, resource 4, ident mtask
# k, 23, x 0.0, resource 8, ident mtask


class TitanSolver(object):
    def __init__(self, method):
        self.method = method

    def job_selection(self, required_resource_list, weight_per_allocation_list, equalivent_allocation_list, unique_job_num, cluster_capacity, max_seconds=1, power=1): 

        m = Model(solver_name = GRB)
        var_len = len(required_resource_list)
        X = [m.add_var(var_type=BINARY) for i in range(var_len)]
        obj_list = [X[i] * (weight_per_allocation_list[i] ** power) for i in range(var_len)]

        if power > 0: 
            m.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list)))) 
        else: 
            m.objective = minimize(xsum(obj_list[i] for i in range(len(obj_list)))) 
        
        for key in cluster_capacity.keys(): 
            resource_list = [sum(required_resource_list[i][key]) if key in required_resource_list[i] else 0 for i in range(var_len)]
            m += xsum(X[i] * resource_list[i] for i in range(var_len)) <= cluster_capacity[key] 

        left, right = 0, 0
        left_list, right_list = list(), list() 
        
        for i in range(unique_job_num): 
            # print('i == {}, {}'.format(i, len(equalivent_allocation_list)))
            for j in range(left, len(equalivent_allocation_list)): 
                if equalivent_allocation_list[j] != i: 
                    right = j 
                    break 
            if equalivent_allocation_list[j] == i: right = j + 1
            m += xsum(X[j] for j in range(left, right)) == 1
            left_list.append(left) 
            right_list.append(right)
            left = right 

        m.optimize(max_seconds=max_seconds)
        allocated_gpu_solution = list() 
        left, right = 0, 0
        for i in range(unique_job_num): 
            allocated = False 
            for j in range(left_list[i], right_list[i]): 
                res = X[j].x
                if res is not None and res > 0.5: 
                    allocated_gpu_solution.append(required_resource_list[j])
                    allocated = True 
            if not allocated: 
                allocated_gpu_solution.append(None) 
        return allocated_gpu_solution
