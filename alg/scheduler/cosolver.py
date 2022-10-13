from mip import *
import numpy as np
import copy
import math

def compute_maximum_lease(expect_maximum_end_time, lease_time_interval, cur_lease_index):
    return math.floor(expect_maximum_end_time / lease_time_interval) - cur_lease_index

class MIPSolver(object):
    def __init__(self, method):
        self.method = method

    # @timeout_decorator.timeout()
    def job_selection(self, required_resource_list, required_block_list, maximum_block_list, resource_num_list, \
        info_reward_list, duration_length_list, virtual_cluster, virtual_node_list, objective, max_seconds=5):
        maximum_block = max(maximum_block_list)
        if len(required_resource_list) == 0 :
            return None, None
        if maximum_block == 0:
            return [None for _ in range(len(required_resource_list))], [None for _ in range(len(required_resource_list))]
        if isinstance(resource_num_list, int):
            resource_num_list = [resource_num_list for _ in range(maximum_block)]
        m = Model(solver_name=GRB)
        var_len = len(required_resource_list) * maximum_block
        X = [m.add_var(var_type=BINARY) for i in range(len(required_resource_list) * maximum_block)]
        S = [m.add_var(var_type=BINARY) for i in range(len(required_resource_list))]
        obj_list = [S[j] * info_reward_list[j].soft_val * info_reward_list[j].duration_val for j in range(len(required_resource_list))]
        
        if objective == 'random':
            obj_list = [S[0]]
        else:
            max_resource_num = max(resource_num_list)
            for i in range(len(required_resource_list)):
                j = i * maximum_block
                obj_list.append(X[j] * required_resource_list[j // maximum_block] / max_resource_num)

        m.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list))))

        # job-wise
        for i in range(len(required_resource_list)):
            m += xsum(X[j] * info_reward_list[i].place_val * duration_length_list[j - i * maximum_block]  for j in range(i * maximum_block, i * maximum_block + maximum_block_list[i])) >= S[i] * required_block_list[i]
            m += xsum(X[j] * info_reward_list[i].place_val * duration_length_list[j - i * maximum_block]  for j in range(i * maximum_block, i * maximum_block + maximum_block_list[i])) <= S[i] * (required_block_list[i] + duration_length_list[maximum_block-1])
            if maximum_block_list[i] <  maximum_block:
                m += xsum(X[j] for j in range(i * maximum_block + maximum_block_list[i], (i+1) * maximum_block)) == 0

        # resource-wise
        for i in range(maximum_block):
            # m += xsum(X[j] * required_resource_list[j // maximum_block] * S[j // maximum_block] for j in range(i, var_len, maximum_block) ) <= resource_num_list[i] * S[j // maximum_block]
            m += xsum(X[j] * required_resource_list[j // maximum_block] for j in range(i, var_len, maximum_block) ) <= resource_num_list[i] #  * S[j // maximum_block]

        #  placement-wise 
        if virtual_cluster is not None:
            print(len(required_resource_list))
            print(len(required_block_list))
            print(len(maximum_block_list))
            print(virtual_cluster)
            print(i, len(virtual_node_list), len(X))
            m += xsum(X[i * maximum_block] * virtual_node_list[i][8] for i in range(len(virtual_node_list)) ) <= virtual_cluster[8]
            m += xsum(X[i * maximum_block] * virtual_node_list[i][4] + 2 * X[i * maximum_block] * virtual_node_list[i][8] for i in range(len(virtual_node_list)) ) <= 2 * virtual_cluster[8] + virtual_cluster[4]
            m += xsum(X[i * maximum_block] * virtual_node_list[i][2] + 2 * X[i * maximum_block] * virtual_node_list[i][4] + 4 * X[i * maximum_block] * virtual_node_list[i][8] for i in range(len(virtual_node_list)) ) <= 4 * virtual_cluster[8] + 2 * virtual_cluster[4]  + virtual_cluster[2]
            m += xsum(X[i * maximum_block] * virtual_node_list[i][1] +  2 * X[i * maximum_block] * virtual_node_list[i][2] +  4 * X[i * maximum_block] * virtual_node_list[i][4] +  8 * X[i * maximum_block] * virtual_node_list[i][8] for i in range(len(virtual_node_list)) ) <= 8 * virtual_cluster[8] + 4 * virtual_cluster[4]  + 2 * virtual_cluster[2] + virtual_cluster[2]

        job_id = -1
        i = 0
        while i < len(info_reward_list):
            if info_reward_list[i] == job_id:
                i += 1
                continue 
            job_id = info_reward_list[i].job_id
            left = i
            right = len(info_reward_list)
            for j in range(i+1, len(info_reward_list)):
                if info_reward_list[j].job_id != job_id:
                    right = j
                    break
            m += xsum(S[j] for j in range(left, right)) <= 1
            i = right
            if right < len(info_reward_list):
                job_id = info_reward_list[right].job_id
        m.optimize(max_seconds=max_seconds, max_seconds_same_incumbent=1)
        solution_matrix = list()
        
        for i in range(len(required_resource_list)):
            start = i * maximum_block
            solution = list()
            for j in range(start, start+maximum_block_list[i]):
                res = X[j].x
                if res is not None:
                    res = 0 if res < 0.5 else 1
                solution.append(res)
            solution_matrix.append(solution)
        soft_matrix = list()
        for i in range(len(info_reward_list)):
            soft_matrix.append(S[i].x)
        return solution_matrix, soft_matrix


    def cal_reward(self, required_resource_list, required_block_list, maximum_block_list, resource_num_list, \
        info_reward_list, duration_length_list, virtual_cluster, virtual_node_list, objective, max_seconds=5):
        maximum_block = max(maximum_block_list)
        if len(required_resource_list) == 0 :
            return None, None
        if maximum_block == 0:
            return [None for _ in range(len(required_resource_list))], [None for _ in range(len(required_resource_list))]
        if isinstance(resource_num_list, int):
            resource_num_list = [resource_num_list for _ in range(maximum_block)]
        m = Model(solver_name=GRB)
        var_len = len(required_resource_list) * maximum_block
        X = [m.add_var(var_type=BINARY) for i in range(len(required_resource_list) * maximum_block)]
        S = [m.add_var(var_type=BINARY) for i in range(len(required_resource_list))]
        obj_list = [S[j] * info_reward_list[j].soft_val * info_reward_list[j].duration_val for j in range(len(required_resource_list))]
        
        if objective == 'random':
            obj_list = [S[0]]

        m.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list))))

        # job-wise
        for i in range(len(required_resource_list)):
            m += xsum(X[j] * info_reward_list[i].place_val * duration_length_list[j - i * maximum_block]  for j in range(i * maximum_block, i * maximum_block + maximum_block_list[i])) >= S[i] * required_block_list[i]
            m += xsum(X[j] * info_reward_list[i].place_val * duration_length_list[j - i * maximum_block]  for j in range(i * maximum_block, i * maximum_block + maximum_block_list[i])) <= S[i] * (required_block_list[i] + duration_length_list[maximum_block-1])
            if maximum_block_list[i] <  maximum_block:
                m += xsum(X[j] for j in range(i * maximum_block + maximum_block_list[i], (i+1) * maximum_block)) == 0

        # resource-wise
        for i in range(maximum_block):
            # m += xsum(X[j] * required_resource_list[j // maximum_block] * S[j // maximum_block] for j in range(i, var_len, maximum_block) ) <= resource_num_list[i] * S[j // maximum_block]
            m += xsum(X[j] * required_resource_list[j // maximum_block] for j in range(i, var_len, maximum_block) ) <= resource_num_list[i] #  * S[j // maximum_block]

        #  placement-wise 
        if virtual_cluster is not None:
            print(len(required_resource_list))
            print(len(required_block_list))
            print(len(maximum_block_list))
            print(virtual_cluster)
            print(i, len(virtual_node_list), len(X))
            m += xsum(X[i * maximum_block] * virtual_node_list[i][8] for i in range(len(virtual_node_list)) ) <= virtual_cluster[8]
            m += xsum(X[i * maximum_block] * virtual_node_list[i][4] + 2 * X[i * maximum_block] * virtual_node_list[i][8] for i in range(len(virtual_node_list)) ) <= 2 * virtual_cluster[8] + virtual_cluster[4]
            m += xsum(X[i * maximum_block] * virtual_node_list[i][2] + 2 * X[i * maximum_block] * virtual_node_list[i][4] + 4 * X[i * maximum_block] * virtual_node_list[i][8] for i in range(len(virtual_node_list)) ) <= 4 * virtual_cluster[8] + 2 * virtual_cluster[4]  + virtual_cluster[2]
            m += xsum(X[i * maximum_block] * virtual_node_list[i][1] +  2 * X[i * maximum_block] * virtual_node_list[i][2] +  4 * X[i * maximum_block] * virtual_node_list[i][4] +  8 * X[i * maximum_block] * virtual_node_list[i][8] for i in range(len(virtual_node_list)) ) <= 8 * virtual_cluster[8] + 4 * virtual_cluster[4]  + 2 * virtual_cluster[2] + virtual_cluster[2]

        job_id = -1
        i = 0
        while i < len(info_reward_list):
            if info_reward_list[i] == job_id:
                i += 1
                continue 
            job_id = info_reward_list[i].job_id
            left = i
            right = len(info_reward_list)
            for j in range(i+1, len(info_reward_list)):
                if info_reward_list[j].job_id != job_id:
                    right = j
                    break
            m += xsum(S[j] for j in range(left, right)) <= 1
            i = right
            if right < len(info_reward_list):
                job_id = info_reward_list[right].job_id
        m.optimize(max_seconds=max_seconds, max_seconds_same_incumbent=1)
        solution_matrix = list()
        
        for i in range(len(required_resource_list)):
            start = i * maximum_block
            solution = list()
            for j in range(start, start+maximum_block_list[i]):
                res = X[j].x
                if res is not None:
                    res = 0 if res < 0.5 else 1
                solution.append(res)
            solution_matrix.append(solution)
        soft_matrix = list()
        for i in range(len(info_reward_list)):
            soft_matrix.append(S[i].x)
        reward_val = 0
        run_job_num = 0
        for j in range(len(required_resource_list)):
            reward_val += S[j].x * info_reward_list[j].soft_val * info_reward_list[j].duration_val
            run_job_num += S[j].x
        return reward_val, run_job_num

