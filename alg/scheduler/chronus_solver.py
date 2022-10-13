from mip import *
import numpy as np
import copy
import math


def compute_maximum_lease(expect_maximum_end_time, lease_time_interval, cur_lease_index):
    return math.floor(expect_maximum_end_time / lease_time_interval) - cur_lease_index
    # return math.ceil(expect_maximum_end_time / lease_time_interval) - cur_lease_index


def compute_emergence(expected_remaining_time, true_remaining_time, required_gpu_num):
    # return -1 * true_remaining_time * required_gpu_num
    # return -expected_remaining_time * required_gpu_num
    # return 1 * true_remaining_time * required_gpu_num
    return -1 * true_remaining_time * required_gpu_num




class SemiPreemptMIPSolver(object):
    def __init__(self, method):
        self.method = method

    def job_selection(self, required_resource_list, required_block_list, maximum_block_list, reward_list, existing_solution, resource_num_list, max_seconds=10):
        if len(maximum_block_list) == 0:
            return list()
        
        maximum_block = max(maximum_block_list)
        solution = list()
        for i in range(maximum_block):
            if i < len(existing_solution): 
                solution.append(existing_solution[i])
            else:
                solution.append(resource_num_list[i])
        if True:
            solution_matrix = list()
            for i in range(len(required_resource_list)):
                sol = None 
                # for j in range(maximum_block_list[i] - 1, required_block_list[i] - 1, -1):
                for j in range(required_block_list[i], maximum_block_list[i]+1):
                    left = j - required_block_list[i]
                    right = left + required_block_list[i]
                    done = True
                    for k in range(left, right):
                        # print(k, left, right, len(solution), maximum_block)
                        if solution[k] < required_resource_list[i]:
                            done = False
                    if done: 
                        sol = [0 for _ in range(maximum_block_list[i])]
                        for k in range(left, right):
                            sol[k] = 1
                            solution[k] -= required_resource_list[i]
                        break
                solution_matrix.append(sol)
            return solution_matrix
        
        
        m = Model(solver_name=GRB)
        X = [m.add_var(var_type=BINARY) for i in range(len(required_resource_list) * maximum_block)]
        obj_list = [X[i] * reward_list[i // maximum_block] for i in range(len(required_resource_list) * maximum_block)]
        m.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list))))

        # job-wise
        for i in range(len(required_resource_list)):
            delta_len = maximum_block_list[i] - required_block_list[i] + 1
            m += xsum(X[j]  for j in range(i * maximum_block, i * maximum_block + delta_len)) <= 1
            
            if delta_len <  maximum_block:
                m += xsum(X[j] for j in range(i * maximum_block + delta_len, (i+1) * maximum_block)) == 0
        
        # resource-wise 
        for i in range(maximum_block):
            resource_list = list() 
            for j in range(len(required_block_list)):
                delta_len = maximum_block_list[j] - required_block_list[j] + 1
                for k in range(delta_len):
                    if k <= i and k + required_block_list[j] - 1 >= i:
                        resource_list.append(X[j * maximum_block + k] * required_resource_list[j]) # TODO
            m += xsum(resource_list) <= solution[i]

        m.optimize(max_seconds=max_seconds)
        solution_matrix = list()

        for i in range(len(required_resource_list)):
            start = i * maximum_block
            sol = list()
            start_idx = -1
            for j in range(start, start+maximum_block_list[i]):
                res = X[j].x
                if res is not None:
                    res = 0 if res < 0.5 else 1
                sol.append(res)
                if res == 1: start_idx = j

            assert sum(sol) <= 1, 'only allowed to select one solution'
            if sum(sol) == 0:
                solution_matrix.append(None)
            else:
                for j in range(start_idx, start_idx + required_block_list[i]):
                    sol[j - start] = 1
                solution_matrix.append(sol)
        
        return solution_matrix



class NoPreemptMIPSolver(object):
    def __init__(self, method):
        self.method = method

    def job_selection(self, required_resource_list, required_block_list, maximum_block_list, reward_list, existing_solution, resource_num_list, max_seconds=5):
        if len(maximum_block_list) == 0:
            return list()
        
        maximum_block = max(maximum_block_list)
        solution = list()
        for i in range(maximum_block):
            if i < len(existing_solution): 
                solution.append(existing_solution[i])
            else:
                solution.append(resource_num_list[i])
        if True:
            solution_matrix = list()
            for i in range(len(required_resource_list)):
                sol = None 
                # for j in range(maximum_block_list[i] - 1, required_block_list[i] - 1, -1):
                for j in range(required_block_list[i], maximum_block_list[i]+1):
                    left = j - required_block_list[i]
                    right = left + required_block_list[i]
                    done = True
                    for k in range(left, right):
                        # print(k, left, right, len(solution), maximum_block)
                        if solution[k] < required_resource_list[i]:
                            done = False
                    if done: 
                        sol = [0 for _ in range(maximum_block_list[i])]
                        for k in range(left, right):
                            sol[k] = 1
                            solution[k] -= required_resource_list[i]
                        break
                solution_matrix.append(sol)
            return solution_matrix
        
        
        m = Model(solver_name=CBC)
        X = [m.add_var(var_type=BINARY) for i in range(len(required_resource_list) * maximum_block)]
        print('number of variables {}'.format(len(X)))
        obj_list = [X[i] * reward_list[i // maximum_block] for i in range(len(required_resource_list) * maximum_block)]
        m.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list))))

        # job-wise
        for i in range(len(required_resource_list)):
            delta_len = maximum_block_list[i] - required_block_list[i] + 1
            m += xsum(X[j]  for j in range(i * maximum_block, i * maximum_block + delta_len)) <= 1
            
            if delta_len <  maximum_block:
                m += xsum(X[j] for j in range(i * maximum_block + delta_len, (i+1) * maximum_block)) == 0
        
        # resource-wise 
        for i in range(maximum_block):
            resource_list = list() 
            for j in range(len(required_block_list)):
                delta_len = maximum_block_list[j] - required_block_list[j] + 1
                for k in range(delta_len):
                    if k <= i and k + required_block_list[j] - 1 >= i:
                        resource_list.append(X[j * maximum_block + k] * required_resource_list[j]) # TODO
            m += xsum(resource_list) <= solution[i]

        m.optimize(max_seconds=max_seconds)
        solution_matrix = list()

        for i in range(len(required_resource_list)):
            start = i * maximum_block
            sol = list()
            start_idx = -1
            for j in range(start, start+maximum_block_list[i]):
                res = X[j].x
                if res is not None:
                    res = 0 if res < 0.5 else 1
                sol.append(res)
                if res == 1: start_idx = j

            assert sum(sol) <= 1, 'only allowed to select one solution'
            if sum(sol) == 0:
                solution_matrix.append(None)
            else:
                for j in range(start_idx, start_idx + required_block_list[i]):
                    sol[j - start] = 1
                solution_matrix.append(sol)
        
        return solution_matrix

