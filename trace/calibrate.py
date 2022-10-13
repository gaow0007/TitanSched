import os, sys
import csv
import yaml 
from easydict import EasyDict
import random 
import numpy as np 
from datetime import datetime
import pandas as pd
import ast 


def time_check(time_info): 
    return time_info >= args.min_time and time_info <= args.max_time # 10 * 3600 * 1

def get_cdf(data):
    """Returns the CDF of the given data.
    
       Args:
           data: A list of numerical values.
           
       Returns:
           An pair of lists (x, y) for plotting the CDF.
    """
    sorted_data = sorted(data)
    p = 100. * np.arange(len(sorted_data)) / (len(sorted_data) - 1)
    return sorted_data, p


def parse_mlass(filename): 
    import math 
    columns = ['start_time', 'runtime_i', 'plan_gpu', 'status']
    df = pd.DataFrame(pd.read_csv(filename), columns=columns).dropna() 
    df = df.loc[df.status=='Terminated']
    df = df.loc[df.plan_gpu > 0]
    df.plan_gpu = df.plan_gpu.apply(lambda c: int(math.ceil(c / 100)))
    df.rename(columns = {'start_time':'submission_time', 'runtime_i':'duration', 'plan_gpu':'num_gpus'}, inplace=True)
    df.drop(columns=['status'], inplace=True)
    return df.reset_index()

def parse_philly(filename): 
    columns = None
    df = pd.DataFrame(pd.read_csv(filename), columns=columns).dropna() 
    date_format = '%Y-%m-%d %H:%M:%S'
    base_time = datetime.strptime('2017-01-01 00:00:00', date_format)
    df.timestamp = df.timestamp.apply(lambda c: (datetime.strptime(c, date_format) - base_time).total_seconds())
    df.rename(columns = {'timestamp':'submission_time'}, inplace=True)
    df.drop(columns=['cluster', 'gpu_time'], inplace=True)
    return df.reset_index()

            # if int(line['gpu_num']) == 0: continue
            # if line['state'] != 'COMPLETED': continue

def parse_helios(filename): 
    columns = ['gpu_num', 'state', 'submit_time', 'duration']
    df = pd.DataFrame(pd.read_csv(filename), columns=columns).dropna() 
    df = df.loc[df.gpu_num > 0]
    df = df.loc[df.state == 'COMPLETED']
    date_format = '%Y-%m-%d %H:%M:%S'
    base_time = datetime.strptime('2020-01-01 00:00:00', date_format)
    df.submit_time = df.submit_time.apply(lambda c: (datetime.strptime(c, date_format) - base_time).total_seconds())
    df.rename(columns = {'gpu_num':'num_gpus', 'submit_time': 'submission_time'}, inplace=True)
    df.drop(columns=['state'], inplace=True)
    return df.reset_index()



def parse_BM(filename): 
    # process_BM(filename)
    # exit(0)
    columns = ['Submit', 'Elapsed', 'AllocGRES', 'State']
    df = pd.DataFrame(pd.read_csv(filename, delimiter='|'), columns=columns).dropna() 
    df.rename(columns = {'Submit':'submission_time', 'AllocGRES': 'num_gpus', 'Elapsed':'duration'}, inplace=True)
    df = df.loc[df.State == 'COMPLETED'] 
    date_format = '%Y-%m-%dT%H:%M:%S'
    base_time = datetime.strptime('2022-01-01T00:00:00', date_format)
    df.submission_time = df.submission_time.apply(lambda c: (datetime.strptime(c, date_format) - base_time).total_seconds())
    df.duration = df.duration.apply(lambda duration: float(duration.split('(')[0]))
    df.num_gpus = df.num_gpus.apply(lambda num_gpus: int(num_gpus.split(':')[1]))
    df = df.loc[df.duration > 0]
    df = df.loc[df.num_gpus > 0]
    df = df.dropna() 
    df.drop(columns=['State'], inplace=True)
    return df.reset_index() 


def process_BM(filename):
    key_list = ['JobID', 'JobName', 'User', 'Submit', 'Start', 'End', 'Elapsed', 'NNodes', 'NCPUS', 'NodeList', \
        'PhxPriority', 'UserPriority', 'Partition', 'Account', 'AllocGRES', 'AllocCPUS', 'State', 'ExitCode']
    with open(filename, 'r') as f: 
        reader = csv.DictReader(f, delimiter='|')
        lines = [line for line in reader]

    with open(filename+'.bak', 'w') as f: 
        f.write('|'.join(key_list))
        f.write('\n')
        for line in lines: 
            val_list = list() 
            for key in key_list: val_list.append(line[key])
            f.write('|'.join(val_list))
            f.write('\n')


def get_context_switch_overhead(placement, pipeline): 
    placement2context_switch_overhead_no_pipeline = {
            1 : 18,
            2: 22,  
            3: 29,  
            4: 29, 
            5: 29, 
            6: 29, 
            7: 29, 
            8: 46, 
        }
    return placement2context_switch_overhead_no_pipeline[sum(placement)]


def get_max_throughput(placement): 
    if sum(placement) == 1: 
        return 1 / 30.883228171955455
    elif sum(placement) == 2: 
        return 1 / 16.883228171955455
    elif sum(placement) <= 4: 
        return 1 / 9.52039733800021
    elif sum(placement) <= 8: 
        return 1 / 5.52039733800021
    else:
        raise NotImplementedError
    

def add_fm_info(sample): 
    # model_types = [np.random.choice(['imagenet', 'cifar10','Oxford Flowers-102', 'Oxford-IIIT-Pets']) for _ in range(args.num_jobs)]
    import string
    import math 
    job_choices = [string.ascii_uppercase[i] for i in range(args.fm_cluster)] 
    job_types = [np.random.choice(job_choices) for _ in range(args.num_jobs)]
    target_batch_list = list() 
    data_scale_list = list() 
    target_iteration_list = list() 
    for idx, job_type in enumerate(job_types): 
        duration = sample['duration'].iloc[idx]
        num_gpus = sample['num_gpus'].iloc[idx]
        phony_duration = duration - get_context_switch_overhead(placement=[num_gpus], pipeline=False)
        target_batch_size = 512
        # if num_gpus >= 4: 
        #     target_batch_size = 512
        # elif num_gpus >= 2: 
        #     target_batch_size = 256
        # elif num_gpus == 1: 
        #     target_batch_size = 128 
        # else:
        #     raise NotImplementedError
        
        target_batch_list.append(target_batch_size)

        phony_iteration = int(math.ceil(phony_duration * get_max_throughput([num_gpus]))) 
        normalized_iteration = phony_iteration * (512 // target_batch_size)
        data_scale = int(math.ceil(np.exp((normalized_iteration - args.p0) / args.p1)))
        # data_scale = exp[(iter - p0) / p1]
        # iter = p0 + p1 * np.log(data_scale)
        

        data_scale_list.append(data_scale)
        target_iteration_list.append(phony_iteration)
    # sample.drop(columns=['num_gpus'], inplace=True)
    # sample.drop(columns=['duration'], inplace=True)
    sample['application'] = job_types
    sample['data_scale'] = data_scale_list 
    sample['target_batch_size'] = target_batch_list
    sample['target_iteration'] = target_iteration_list



def add_ddl_time(sample):
    job_types = [np.random.choice(['best', 'soft', 'strict'], p = [args.best_effort, args.soft_SLO, args.strict_SLO]) for _ in range(args.num_jobs)]
    sample['ddl_type'] = job_types
    ddl_times, ddl_values = list(), list() 
    for idx, job_type in enumerate(job_types): 
        # import pdb; pdb.set_trace() 
        submission_time = sample['submission_time'].iloc[idx]
        duration = sample['duration'].iloc[idx]
        if job_type == 'best': 
            expect_time_list = [int(submission_time)]
            expect_value_list = [1]
        elif job_type == 'strict' or job_type == 'soft': 
            expect_time = submission_time + random.randint(int(1.2 * duration), min(int(2 * duration), 14 * 24 * 60 * 60))
            expect_value_list = [100]
            submit_day, submit_hour, submit_minute = compute_date(submission_time)
            expect_day, expect_hour, expect_minute = compute_date(expect_time)
            if submit_hour == expect_hour: 
                expect_hour += 1
            expect_time = expect_day * 24 * 60 * 60 + expect_hour * 60 * 60  + expect_minute * 60 
            expect_time_list = [int(expect_time)]
            if job_type == 'soft': 
                for scale, value in zip([1.1, 1.2, 1.5], [80, 50, 20]):
                    expect_time = (expect_time_list[0] - submission_time) * scale + submission_time
                    expect_day, expect_hour, expect_minute = compute_date(expect_time)
                    expect_time = expect_day * 24 * 60 * 60 + expect_hour * 60 * 60  + expect_minute * 60 
                    expect_value_list.append(value)
                    expect_time_list.append(int(expect_time))
        else: 
            raise NotImplementedError
            
        ddl_time = '-'.join([str(int(item)) for item in expect_time_list])
        ddl_value = '-'.join([str(item) for item in expect_value_list])
        ddl_times.append(ddl_time)
        ddl_values.append(ddl_value)
    sample['ddl_time_list'] = ddl_times 
    sample['ddl_value_list'] = ddl_values

def compute_date(cur_time):
    # print(cur_time)
    day = cur_time // (24 * 60 * 60)
    hour = (cur_time % (24*60*60)) // (60 * 60)
    minute = cur_time % (60*60) // 60
    return day, hour, minute



def full_main(): 
    mlaas_trace = parse_mlass('full_trace/MLaaS.csv')
    philly_trace = parse_philly('full_trace/Philly.csv')
    helios_trace = parse_helios('full_trace/Helios.csv')
    bm_trace = parse_BM('full_trace/BM.csv')
    rng = random.Random(args.seed)
    for trace_name, trace in [('MLaas', mlaas_trace), ('Philly', philly_trace), ('Helios', helios_trace), ('BM', bm_trace)]: 
        sample = trace
        
        sample.rename(columns = {'index':'name'}, inplace=True)

        for key in ['submission_time', 'num_gpus', 'duration']: 
            if hasattr(sample, key): 
                sample = sample.astype({key: 'int32'}) 
        sample.set_index("name").to_csv('{}/{}.csv'.format(args.save_root, trace_name))


def main(): 
    mlaas_trace = parse_mlass('full_trace/MLaaS.csv')
    philly_trace = parse_philly('full_trace/Philly.csv')
    helios_trace = parse_helios('full_trace/Helios.csv')
    bm_trace = parse_BM('full_trace/BM.csv')
    rng = random.Random(args.seed)
    for trace_name, trace in [('MLaas', mlaas_trace), ('Philly', philly_trace), ('Helios', helios_trace), ('BM', bm_trace)]: 
        sample = trace.loc[(trace.duration >= args.min_time) & (trace.duration < args.max_time)]
        sample.insert(1, 'gpu_time', sample.num_gpus * sample.duration, True)
        if args.gpu_limit: 
            sample.num_gpus = sample.num_gpus.apply(lambda gpu: min(gpu, args.gpu_limit))
        else: 
            sample.num_gpus = sample.num_gpus.apply(lambda gpu: min(gpu, 64))
        
        sample = sample.loc[sample.gpu_time < 100 * 3600]
        if args.num_jobs > 0: 
            sample = sample.sample(n=args.num_jobs, random_state=rng.randint(0, 1 << 32))

        sample.submission_time = sample.submission_time.apply(lambda st : st % (24 * 60 * 60))
        if args.add_ddl: 
            add_ddl_time(sample)
        
        if args.add_fm: 
            add_fm_info(sample)

        sample.rename(columns = {'index':'name'}, inplace=True)
        sample.rename(columns = {'index':'name'}, inplace=True)

        for key in ['submission_time', 'num_gpus', 'duration']: 
            if hasattr(sample, key): 
                sample = sample.astype({key: 'int32'}) 
        
    
        sample.drop(columns=['gpu_time'], inplace=True)
        sample.set_index("name").to_csv('{}/{}.csv'.format(args.save_root, trace_name))
        # print(trace_name, sample.groupby(["num_gpus"]).size())


def get_config_from_yaml(yaml_file):
    """
    Get the config from a yaml file
    :param string yaml_file: yaml configuration file
    :return: EasyDict config
    """
    with open(yaml_file) as fp:
        config_dict = yaml.safe_load(fp)
    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)
    return config

def add_by_yaml(yaml): 
    assert yaml.endswith('yaml')
    config = get_config_from_yaml(yaml)
    for key, value in config.items(): 
        if not hasattr(args, key): 
            setattr(args, key, value)
        elif getattr(args, key) != value: 
            setattr(args, key, value)


# python calibrate.py --min_time=300 --max_time=36000 --num_jobs=320
import argparse
parser = argparse.ArgumentParser(description='Calibrate GPU Cluster Trace')
parser.add_argument('--min_time', default=300, type=int, required=True, help='min duration')
parser.add_argument('--max_time', default=10*3600, type=int, required=True, help='max duration')
parser.add_argument('--add_user', default=False, type=ast.literal_eval, required=True, help='add user attribute')
parser.add_argument('--add_job_name', default=False, type=ast.literal_eval, required=True, help='add job name attribute')
parser.add_argument('--add_ddl',  default=False, type=ast.literal_eval, required=True, help='add ddl for each job')
parser.add_argument('--ddl_yaml', default=None, type=str, required=False, help='ddl configurations')
parser.add_argument('--add_fm',  default=False, type=ast.literal_eval, required=True, help='add ddl for each job')
parser.add_argument('--fm_yaml', default=None, type=str, required=False, help='fm configurations')
parser.add_argument('--add_norm',  default=False, type=ast.literal_eval, required=True, help='add ddl for each job')
parser.add_argument('--norm_yaml', default=None, type=str, required=False, help='norm configurations')
parser.add_argument('--num_jobs', default=160, type=int, help='num jobs')
parser.add_argument('--save_root', default=None, type=str, help='sample trace save path')
parser.add_argument('--seed', default=-1, type=int, help='seed')
parser.add_argument('--full_trace', default=False, type=ast.literal_eval, help='whether extract all trace')
args = parser.parse_args()



if __name__ == '__main__': 
    if args.full_trace: 
        if not os.path.exists(args.save_root): 
            os.makedirs(args.save_root) 
        full_main() 
    else: 
        if args.add_ddl: 
            add_by_yaml(args.ddl_yaml)
        if args.add_fm: 
            add_by_yaml(args.fm_yaml)
        if args.add_norm: 
            add_by_yaml(args.norm_yaml)
        if args.save_root is None: 
            args.save_root = os.path.join('./min-{}-max-{}-num-{}'.format(args.min_time, args.max_time, args.num_jobs if args.num_jobs > 0 else 'full'))
        if not os.path.exists(args.save_root): 
            os.makedirs(args.save_root) 
        main()
