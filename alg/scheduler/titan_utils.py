import collections
from client.application.foundation_model import TaskScale

METHOD="FAIR"

def append(job, job_cluster): 
    if job.application.name not in job_cluster: 
        job_cluster[job.application.name] = [job]
    else: 
        job_cluster[job.application.name].append(job)

def job_application_priority(jobA): 
    return jobA.max_progress

def application_priority(appA): 
    return TaskScale[appA.split('@')[-1]] 

def compute_weight_metric(METHOD, job, placement, fair_placement, fair_remaining_time, cur_time, scheduling_time_interval): 
    if METHOD == "TIME":
        predict_remaing_time = job.predict_remaining_time(placement)
        # weight = fair_remaining_time / predict_remaing_time
        weight = 1.0 / (predict_remaing_time + 1e-3) # * gpu_num
    elif METHOD == "THR": 
        throughput = job.throughput_estimation(placement, batch_size=job.batch_size)
        # step_time = job.application.get_throughput(placement=placement, local_bsz=32)
        if isinstance(step_time, (tuple, np.ndarray)): 
            step_time = step_time[0]
        weight = 32. * sum(placement) / step_time / (job.max_progress - job.progress)
    elif METHOD == "FAIR": 
        predict_remaing_time = max(job.predict_remaining_time(placement), scheduling_time_interval)
        weight = 1.0 * fair_remaining_time / predict_remaing_time 
    elif METHOD == "FFT": 
        if hasattr(job, 'deserved_service') and job.deserved_service > 0:
            t_remaining = job.predict_remaining_time(placement)
            t_isolated = job.running_time * (job.attained_service / job.deserved_service) + (job.staying_time - job.running_time)
            weight = (job.staying_time) / (t_isolated + t_remaining)
        else: 
            predict_remaing_time = max(job.predict_remaining_time(placement), scheduling_time_interval)
            weight = 1.0 * fair_remaining_time / predict_remaing_time 
        # predict_remaing_time = max(job.predict_remaining_time(placement), scheduling_time_interval)
        # weight = 1.0 * (predict_remaing_time + cur_time - job.submission_time) / (fair_remaining_time + cur_time - job.submission_time)
    return weight


def allocation2num(allocations): 
    if allocations is None: 
        return 0 
    elif isinstance(allocations, int): 
        return allocations 
    elif isinstance(allocations, dict): 
        return sum([sum(placement) for placement in allocations.values()])
    elif isinstance(allocations, collections.Iterable): 
        return sum([gpu for gpu in allocations])
    else: 
        raise NotImplementedError 
    

def convert_allocation(allocation): 
    if allocation is None: 
        return {'V100':0, 'A100':0}
    elif isinstance(allocation, int): 
        return {'V100': allocation, 'A100': 0}
    elif isinstance(allocation, tuple): 
        return {'V100': sum(allocation), 'A100': 0}
    elif isinstance(allocation, list): 
        return {'V100': sum(allocation), 'A100': 0}
    elif isinstance(allocation, dict): 
        assert 'V100' in allocation or 'A100' in allocation
        allocation_dict = dict() 
        for key in ['V100', 'A100']: 
            if key in allocation: 
                allocation_dict[key] = allocation2num(allocation[key])
            else: 
                allocation_dict[key] = 0 
        return allocation_dict
    
    raise NotImplementedError 


def allocation_equal(allocation1, allocation2): 
    allocation1 = convert_allocation(allocation1)
    allocation2 = convert_allocation(allocation2)
    for key in ['V100', 'A100']: 
        if allocation1[key] != allocation2[key]: 
            return False 
    return True 

def build_placement_from_num(num_gpus): 
    if num_gpus == 0: 
        return (0,)
    placement = [4 for _ in range(num_gpus // 4)]
    if num_gpus % 4: placement.append(num_gpus % 4)
    placement = tuple(placement)
    return placement


def create_candidate_allocations(cluster_manager, cluster_gpu_info, heterogeneity=False): 
    candidate_allocations = list() 
    if heterogeneity: 
        for GPU_KIND in ["V100", "A100"]: 
            total_gpu_num = cluster_gpu_info[GPU_KIND] 
            if total_gpu_num == 0: continue 
            for num_gpus in [0, 1, 2, 3] + [4 * i for i in range(1, total_gpu_num//4+1)]: 
                placement = build_placement_from_num(num_gpus)
                candidate_allocations.append({GPU_KIND:placement})
        
        for num_gpus_V100 in [4 * i for i in range(1, cluster_gpu_info['V100'] // 4 + 1)]: 
            placement_V100 = build_placement_from_num(num_gpus_V100)
            for num_gpus_A100 in [4 * i for i in range(1, cluster_gpu_info['A100'] // 4 + 1)]: 
                placement_A100 = build_placement_from_num(num_gpus_A100)
                candidate_allocations.append({"V100": placement_V100, "A100": placement_A100})
    else: 
        candidate_allocations = list() 
        for GPU_KIND in ["V100", "A100"]:
            total_gpu_num = cluster_gpu_info[GPU_KIND]    
            if total_gpu_num == 0: continue 
            for num_gpus in [0, 1, 2, 3] + [4 * i for i in range(1, total_gpu_num//4+1)]: 
                placement = build_placement_from_num(num_gpus)
                candidate_allocations.append({GPU_KIND:placement})
    return candidate_allocations