from heapq import merge
from mip import *
import numpy as np
import copy
import math



class TitanMultiTaskAdaptivitySolver(object):
    def __init__(self, method):
        self.method = method

    def job_selection(self, required_resource_list, weight_per_allocation_list, equalivent_allocation_list, unique_job_num, 
                            merge_required_resource_list, merge_weight_per_allication_list, merge_equalivent_allocation_list, merge_unique_job_num, cluster_capacity, max_seconds=1): 

        m = Model(solver_name = CBC)
        var_len = len(required_resource_list) + len(merge_required_resource_list)
        X = [m.add_var(var_type=BINARY) for i in range(var_len)]
        tot_weight_list = weight_per_allocation_list + merge_weight_per_allication_list
        obj_list = [X[i] * tot_weight_list[i] for i in range(var_len)]
        m.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list)))) 
        tot_resource_list = required_resource_list + merge_required_resource_list
        m += xsum(X[i] * tot_resource_list[i] for i in range(var_len)) <= cluster_capacity

        # single-task intra resource allocation constraint
        left, right = 0, 0
        left_list, right_list = list(), list() 
        for i in range(unique_job_num): 
            reference_allocation = equalivent_allocation_list[left]
            for j in range(left, len(equalivent_allocation_list)): 
                if equalivent_allocation_list[j] != reference_allocation: 
                    right = j 
                    break 
            if equalivent_allocation_list[j] == i: right = j + 1 
            m += xsum(X[j] for j in range(left, right)) <= 1
            left_list.append(left) 
            right_list.append(right)
            left = right 

        # multi-task intra resource allocation constraint
        left, right = 0, 0 
        merge_left_list, merge_right_list = list(), list() 
        for i in range(merge_unique_job_num): 
            reference_allocation = merge_equalivent_allocation_list[left]
            for j in range(left, len(merge_equalivent_allocation_list)): 
                if merge_equalivent_allocation_list[j] != reference_allocation: 
                    right = j
                    break 

            if equalivent_allocation_list[j] == i: right = j + 1 
            m += xsum(X[j + len(required_resource_list)] for j in range(left, right)) <= 1 # be careful about index shift 
            merge_left_list.append(left)
            merge_right_list.append(right)
            left = right 

        # single-task and mult-task resource allocation constraint
        for i in range(merge_unique_job_num): 
            
            merge_idx_list = merge_equalivent_allocation_list[merge_left_list[i]]
            conflict_list = list() 
            for merge_idx in merge_idx_list: 
                left, right = left_list[merge_idx], right_list[merge_idx]
                for j in range(left, right): conflict_list.append(j)
            left, right = merge_left_list[i], merge_right_list[i]
            for j in range(left, right): conflict_list.append(len(required_resource_list) + j) 
            m += xsum(X[j] for j in conflict_list) <= 1


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
        for i in range(merge_unique_job_num): 
            allocated = False 
            for j in range(merge_left_list[i], merge_right_list[i]): 
                res = X[j + len(required_resource_list)].x 
                if res is not None and res > 0.5: 
                    multi_task = True 
                    allocated_gpu_solution.append(merge_required_resource_list[j])
                    allocated = True 
            if not allocated: 
                allocated_gpu_solution.append(0) 
        if multi_task: 
            print("allocated_gpu_solution is {}".format(allocated_gpu_solution))
            # import pdb; pdb.set_trace() 

        return allocated_gpu_solution



class TitanSolver(object):
    def __init__(self, method):
        self.method = method

    def job_selection(self, required_resource_list, weight_per_allocation_list, equalivent_allocation_list, unique_job_num, cluster_capacity, max_seconds=1, power=1): 

        m = Model(solver_name = CBC)
        var_len = len(required_resource_list)
        X = [m.add_var(var_type=BINARY) for i in range(var_len)]
        obj_list = [X[i] * (weight_per_allocation_list[i] ** power) for i in range(var_len)]

        if power > 0: 
            m.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list)))) 
        else: 
            m.objective = minimize(xsum(obj_list[i] for i in range(len(obj_list)))) 

        m += xsum(X[i] * required_resource_list[i] for i in range(var_len)) <= cluster_capacity
        left, right = 0, 0
        left_list, right_list = list(), list() 
        for i in range(unique_job_num): 
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
                allocated_gpu_solution.append(0) 
        return allocated_gpu_solution
