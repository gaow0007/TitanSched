import os, sys
import copy 
import numpy as np 
from collections import OrderedDict
import pymoo 
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.operators.crossover.util import crossover_mask
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

sys.path.insert(0, os.path.basename(__file__) + os.sep + '..' + os.sep + '..') 
from client.job.state import JobState
from .base import BaseScheduler
from .schedutils import resource_summary, schedule_summary, NodeInfo, JobInfo
from .pollux_solver import Problem, Crossover, Mutation, Repair


class PolluxScheduler(BaseScheduler):
    def __init__(self, job_manager, cluster_manager, user_manager, placement, name, logger, **kwargs):
        super(PolluxScheduler, self).__init__(job_manager, cluster_manager, user_manager, placement=placement, name=name, logger=logger)
        assert self.name == 'pollux'
        self.pending_jobs = job_manager.pending_jobs
        self.running_jobs = job_manager.running_jobs
        self.event_jobs = job_manager.event_jobs
        self.completion_jobs = list()
        self.scheduling_time_interval = kwargs.get('scheduling_time_interval', 10)
        self.save_dir = kwargs.get('save_dir', 'result/') 
        
        self._prev_states = None
        self._prev_jobs = None
        self._prev_nodes = None
        

    # abstract method
    def check_resource(self, **kwargs):
        resource_name = kwargs.get('resource', 'gpu')
        assert isinstance(resource_name, str)
        if 'gpu' == resource_name:
            return self.CLUSTER.check_free_gpus()
        if 'cpu' == resource_name:
            return self.CLUSTER.check_free_cpus()
        
        raise NotImplementedError
    

    def try_allocate_resoure(self, job):
        if self.placement.name in ['random', 'consolidate', 'gandiva', 'local_search', 'consolidate_random']:
            ret = self.placement.place_jobs(job)
        else:
            raise NotImplementedError
        return ret


    def release_job_resource(self, job, status=JobState.END):
        if self.placement.name == 'gandiva':
            ret = self.cluster_manager.release_gandiva_job_resource(job, status)
        else:
            ret = self.cluster_manager.release_job_resource(job, status)
        job.release_resource()
        return ret
        

    def move_to_pending(self, job):
        ''' job gets into the system: pending or running, and finally END'''
        job.last_check_time = job.submission_time 
        self.pending_jobs.append(job)
        job.status = JobState.PENDING

    
    def execute_start_job(self, start_job, cur_time):
        self.pending_jobs.append(start_job)
        start_job.status = JobState.PENDING
        self.logger.info('---- job[{}] is added  at time[{}]'.format(start_job.name, cur_time))


    def execute_preempt(self, job, cur_time):
        assert self.release_job_resource(job, status=JobState.PENDING) == True
        job.release_resource() 
        self.logger.info('running job {} is preempted at time {}'.format(job.name, cur_time))
    

    def flush_event_jobs(self, prev_time, cur_time):
        event_list = list()
        for event_job in self.event_jobs:
            if event_job.submission_time <= cur_time:
                assert event_job.submission_time >= prev_time
                event_list.append(event_job)
        
        submit_job_num = 0
        for event_job in event_list: 
            self.execute_start_job(event_job, cur_time)
            self.event_jobs.remove(event_job)
            submit_job_num += 1
        self.submit_job_num_list.append(submit_job_num)


    def flush_running_jobs(self, prev_time, cur_time):
        self.logger.info("---------------- SIMULATOR TIME: {} ----------------".format(cur_time))
        self.logger.info("Active jobs:")
        used_gpus = 0
        need_remove_jobs = list()
        for job in self.running_jobs:
            job.step(cur_time - max(prev_time, job.submission_time))
            if job.completion_time is not None: 
                assert self.release_job_resource(job) == True
                job.status = JobState.END
                need_remove_jobs.append(job)
                self.completion_jobs.append(job)
                self.logger.info("    {}:\t[submission {}]\t[completition {}]".format(
                      job.name, job.submission_time, job.completion_time))
            else:
                used_gpus += sum(job.placement)
                batch_size = job.atomic_bsz * (job.accum_steps + 1) * sum(job.placement)
                self.logger.info("    {}:\t[epoch {}]\t[restarts {}]\t[batch size {}]\t[target {}]\t[placement {}]\t[progress {:.02f}%]".format(
                      job.name, job.epoch, job.num_restarts, batch_size, job.target_batch_size, job.placement, 100 * job.progress / job.max_progress))
        self.logger.info("GPU utilization: {}".format(used_gpus))
        for job in need_remove_jobs:
            self.running_jobs.remove(job)


    def flush_pending_jobs(self, prev_time, cur_time):
        for job in self.pending_jobs:
            job.step(seconds=cur_time - max(prev_time, job.submission_time))

    def get_node_infos(self, ): 
        return {
            idx: NodeInfo({"nvidia.com/gpu": self.cluster_manager.num_gpu_p_node}, preemptible=False)
            for idx in range(self.cluster_manager.num_node)
        }
    
    def get_job_infos(self, jobs): 
        job_infos = dict() 
        for job in jobs: 
            job_infos[job.name] = JobInfo(
                resources={"nvidia.com/gpu": 1},
                speedup_fn=job.get_speedup_fn(),
                creation_timestamp=job.submission_time,
                attained_service=job.attained_service,
                min_replicas=0,
                max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
                                job.application.max_batch_size // job.application.min_local_bsz),
                preemptible=True,
                max_progress=job.max_progress - job.progress,
                name=job.name
            )
            # print(job_infos[job.name].max_profiled_replicas)
            job_infos[job.name].num_restarts = job.num_restarts or 0
            job_infos[job.name].age = self.cur_time - job.submission_time
        return job_infos



    def _allocations_to_state(self, allocations, jobs, nodes):
        jobs_index = {key: idx for idx, key in enumerate(jobs)}
        nodes_index = {key: idx for idx, key in enumerate(nodes)}
        state = np.zeros((len(jobs), len(nodes)), dtype=np.int)
        for job_key, alloc in allocations.items():
            for node_key in (key for key in alloc if key in nodes_index):
                state[jobs_index[job_key], nodes_index[node_key]] += 1
        return state

    def _state_to_allocations(self, state, jobs, nodes):
        allocations = {}
        for job_idx, job_key in enumerate(jobs):
            for node_idx, node_key in enumerate(nodes):
                count = state[job_idx, node_idx]
                allocations.setdefault(job_key, []).extend([node_key] * count)
        return allocations

    def _adapt_prev_states(self, jobs, nodes):
        # Adapt the previously saved optimization states to initialize the
        # current genetic algorithm states.
        #shape = (len(self._prev_states), len(jobs), 2 * len(nodes))
        shape = (len(self._prev_states), len(jobs), len(nodes))
        states = np.zeros(shape, dtype=np.int)
        jobs_src = [i for i, key in enumerate(self._prev_jobs) if key in jobs]
        jobs_dst = [i for i, key in enumerate(jobs) if key in self._prev_jobs]
        placeholder = len(self._prev_nodes)  # Next placeholder node to copy.
        # Set allocations for physical (non-placeholder) nodes.
        nodes_index = {key: i for i, key in enumerate(self._prev_nodes)}
        for i, key in enumerate(nodes):
            if key in nodes_index:
                states[:, jobs_dst, i] = \
                    self._prev_states[:, jobs_src, nodes_index[key]]
            elif placeholder < self._prev_states.shape[2]:
                # New node, use allocations for a previous placeholder node.
                states[:, jobs_dst, i] = \
                    self._prev_states[:, jobs_src, placeholder]
                placeholder += 1
        # Set allocations for placeholder nodes.
        #for i in range(len(nodes), 2 * len(nodes)):
        #    if placeholder < self._prev_states.shape[2]:
        #        states[:, jobs_dst, i] = \
        #            self._prev_states[:, jobs_src, placeholder]
        #        placeholder += 1
        return states

    def _select_result(self, values, max_nodes):
        if np.amin(values[:, 1]) > max_nodes:
            return None
        return np.argmin(np.where(values[:, 1] <= max_nodes, values[:, 0], 0))

    def _desired_nodes(self, utilities, values, nodes):
        idx = self._select_result(values, len(nodes))
        if idx is not None and \
                self._min_util <= utilities[idx] <= self._max_util:
            return len(nodes)
        target_util = (self._min_util + self._max_util) / 2
        best_util = np.inf
        best_val = 0.0
        best_nodes = len(nodes)
        for util, (val, num_nodes) in zip(utilities, values):
            if util > best_util and val < best_val:
                best_util = util
                best_val = val
                best_nodes = num_nodes
            elif util < best_util and val > best_val:
                continue
            elif abs(util - target_util) < abs(best_util - target_util):
                best_util = util
                best_val = val
                best_nodes = num_nodes
        return int(best_nodes)


    def flush_runnable_jobs(self, prev_time, cur_time):
        if len(self.running_jobs) + len(self.pending_jobs) == 0:
            return 
        base_allocations = dict() 
        def ispinned(key, job):
            return not job.preemptible and base_allocations.get(key, []) != []

        jobs = self.get_job_infos(self.running_jobs + self.pending_jobs)
        jobs = OrderedDict(sorted(jobs.items(),
                                  key=lambda kv: (not ispinned(kv[0], kv[1]),
                                                  kv[1].attained_service,
                                                  kv[1].creation_timestamp)))
        nodes = self.get_node_infos() 
        base_state = \
            self._allocations_to_state(base_allocations, jobs, nodes)

        if self._prev_states is None:
            states = np.expand_dims(base_state, 0)
        else:
            states = self._adapt_prev_states(jobs, nodes)
        problem = Problem(list(jobs.values()), list(nodes.values()), base_state)
                          #len(nodes) * [node_template], base_state)
        algorithm = NSGA2(
            pop_size=50,
            # pymoo expects a flattened 2-D array.
            sampling=states.reshape(states.shape[0], -1),
            crossover=Crossover(),
            mutation=Mutation(),
            repair=Repair(),
        )
        result = pymoo.optimize.minimize(problem, algorithm, ("n_gen", 10))
        #states = result.X.reshape(result.X.shape[0], len(jobs), 2 * len(nodes))
        states = result.X.reshape(result.X.shape[0], len(jobs), len(nodes))
        self._prev_states = copy.deepcopy(states)
        self._prev_jobs = copy.deepcopy(jobs)
        self._prev_nodes = copy.deepcopy(nodes)
        # Get the pareto front.
        nds = NonDominatedSorting().do(result.F, only_non_dominated_front=True)
        states = states[nds]
        values = result.F[nds]
        # Construct return values.
        utilities = problem.get_cluster_utilities(states)
        #desired_nodes = self._desired_nodes(utilities, values, nodes)
        desired_nodes = len(nodes)
        idx = self._select_result(values, min(len(nodes), desired_nodes))
        idx = np.argmin(values[:,0])
        self.logger.info("\n" + "-" * 80)
        for i, state in enumerate(states):
            out = "Solution {}:\n".format(i)
            out += "{}\n".format(state)
            out += "Value: {}\n".format(values[i].tolist())
            out += "Utility: {}\n".format(utilities[i])
            out += "Desired nodes: {}\n".format(desired_nodes)
            out += "-" * 80
            # self.logger.info(out)
        self.logger.info("Selected solution %d", idx)
        
        allocations = (self._state_to_allocations(states[idx], jobs, nodes)
                if idx is not None else {})
        self.post_process(allocations, prev_time, cur_time)

    def post_process(self, allocations, prev_time, cur_time): 
        should_run_jobs = list() 
        runnable_jobs = self.running_jobs + self.pending_jobs 
        num_replicas = [0 for job in runnable_jobs]
        for job_idx, job in enumerate(runnable_jobs): 
            allocation = allocations.get(job.name, None)
            if allocation is None: 
                num_replicas[job_idx] = 0 
            else: 
                num_replicas[job_idx] = len(allocation)


        for job_idx, job in enumerate(runnable_jobs): 
            if num_replicas[job_idx] != job.target_num_gpus: 
                # self.logger.info('job {} is preempted at time {}'.format(job['job_id'], cur_time))
                if job.status == JobState.RUNNING:
                    self.execute_preempt(job, cur_time)

                if job in self.running_jobs:
                    self.running_jobs.remove(job)
                if job not in self.pending_jobs:
                    self.pending_jobs.append(job)
            
            if num_replicas[job_idx] > 0: 
                if num_replicas[job_idx] != job.target_num_gpus:
                    should_run_jobs.append(job)
                    job.target_num_gpus = num_replicas[job_idx]
            else: 
                job.target_num_gpus = 0

        self.place_jobs(should_run_jobs, cur_time)

        for job in should_run_jobs: 
            if job.placement is not None: 
                job.status = JobState.RUNNING
            else: 
                job.status = JobState.PENDING

            if job.status == JobState.RUNNING:
                job_idx = runnable_jobs.index(job)
                job.target_num_gpus = num_replicas[job_idx]
            else: 
                job.target_num_gpus = 0

        for job in runnable_jobs: 
            if job.status == JobState.RUNNING:
                if job in self.pending_jobs: 
                    self.pending_jobs.remove(job)
                if job not in self.running_jobs: 
                    self.running_jobs.append(job)

            if job.status == JobState.PENDING:
                if job in self.running_jobs: 
                    self.running_jobs.remove(job)
                if job not in self.pending_jobs: 
                    self.pending_jobs.append(job)
        # import pdb; pdb.set_trace() 
        

    # abstract method
    def place_jobs(self, jobs, cur_time):
        jobs = sorted(jobs, key=lambda e: -e.target_num_gpus)
        for job in jobs:
            if not self.try_allocate_resoure(job): # TODO, make it allowed allocate across nodes
                continue
            
            self.logger.info('pending job {} is resumed at time {}, and allocated {} gpu'.format(job.name, cur_time, job.target_num_gpus))



    def finish_all_jobs(self, ):
        # print('running {}, event {}, pending {}'.format(len(self.running_jobs), len(self.event_jobs), len(self.pending_jobs)))
        return len(self.running_jobs) + len(self.event_jobs) + len(self.pending_jobs) == 0


    def flush_jobs(self, prev_time, cur_time, status):
        if status == JobState.EVENT:
            self.flush_event_jobs(prev_time, cur_time)
        
        elif status == JobState.RUNNING:
            self.flush_running_jobs(prev_time, cur_time)
        
        elif status == JobState.PENDING:
            self.flush_pending_jobs(prev_time, cur_time)
        
        elif status == JobState.RUNNABLE:
            self.flush_runnable_jobs(prev_time, cur_time)

        else:
            raise NotImplementedError

    def run(self, ):
        cur_time = 0
        self.cur_time = cur_time
        while not self.finish_all_jobs():
            prev_time = max(0, cur_time - self.scheduling_time_interval)
            self.flush_jobs(prev_time, cur_time, status=JobState.RUNNING)
            self.flush_jobs(prev_time, cur_time, status=JobState.EVENT)
            self.flush_jobs(prev_time, cur_time, status=JobState.PENDING)
            self.flush_jobs(prev_time, cur_time, status=JobState.RUNNABLE)
            cur_time += self.scheduling_time_interval
            resource_summary(self)
            self.cur_time = cur_time 

        schedule_summary(self)