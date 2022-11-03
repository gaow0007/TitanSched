import os, sys
import csv
import ast 
import yaml 
import random 
import numpy as np 
import pandas as pd
from datetime import datetime
from easydict import EasyDict
sys.path.insert(0, './')
from client.application.foundation_model import FOUNDATIONMODELAPPLICATIONS 
seed=42

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


def main(model, APPLICATIONS): 
    # mlaas_trace = parse_mlass('full_trace/MLaaS.csv')
    # philly_trace = parse_philly('full_trace/Philly.csv')
    # helios_trace = parse_helios('full_trace/Helios.csv')
    bm_trace = parse_BM('trace/full_trace/BM.csv')
    rng = random.Random(seed)
    rng2 = random.Random(seed + 3)
    rng3 = random.Random(seed + 4)
    job_rng = np.random.default_rng(seed)
    from client.application.foundation_model import FMStats
    TRIAL_CNT = 27
    
    for trace_name, trace in [('BM', bm_trace)]: 
        records = list() 
        for application_id in range(args.num_jobs // TRIAL_CNT): 
            appname = list(APPLICATIONS.keys())[::-1][application_id]
            cnt = 0 
            
            for lr in ['1e-5', '2e-5', '4e-5', '1e-4', '4e-4', '1e-3', '1e-2']: 
                for gradient_steps in [1, 4, 8, 12]: 
                    row = dict() 
                    row["submission_time"] = 0 
                    row["target_lr"] = lr 
                    row["target_gradient_steps"] = gradient_steps
                    row["num_gpus"] = 1
                    row["application"] = appname
                    row["name"] =  "{}-{}".format(row["application"], cnt)

                    if 'vit' in model: 
                        target_batch_size = 12 * 4
                    elif 'roberta' in model: 
                        target_batch_size = 4 * 4
                    row["target_batch_size"] = target_batch_size * gradient_steps

                    cnt += 1
                    records.append(row)
                    if cnt == TRIAL_CNT: 
                        break 
                if cnt == TRIAL_CNT: 
                    break 
        
        records.sort(key=lambda v: v["submission_time"])
        for idx, rec in enumerate(records):
            rec["name"] = "{}-{}".format(rec["application"], idx)
    if args.add_metric: 
        colums = ("name", "submission_time", "application", "num_gpus", "target_batch_size", "target_lr", "target_gradient_steps", "target_metric")
    else: 
        colums = ("name", "submission_time", "application", "num_gpus", "target_batch_size")
    colums = ("name", "submission_time", "application", "num_gpus", "target_lr", "target_gradient_steps",  "target_batch_size")
    return pd.DataFrame(records, columns=colums)

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
parser.add_argument('--model', default=None, type=str, help='model name')
parser.add_argument('--seed', default=-1, type=int, help='seed')
parser.add_argument('--repeat_number', default=1, type=int, help='the number of repeat experiments')
parser.add_argument('--full_trace', default=False, type=ast.literal_eval, help='whether extract all trace')
parser.add_argument('--add_metric', default=False, type=ast.literal_eval, help='whether extract all trace')
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
        
        # for model in ['vit', 'vit-large', 'roberta-base', 'roberta-large']: 
        for model_name in [args.model]: # , 'vit', 'vit-large']: 
            APPLICATIONS = {} 
            for task in FOUNDATIONMODELAPPLICATIONS.keys(): 
                if model_name in task: 
                    APPLICATIONS[task] = FOUNDATIONMODELAPPLICATIONS[task]
            for repeat_id in range(args.repeat_number): 

                workload = main(model_name, APPLICATIONS)
                csv = workload.set_index("name").to_csv(os.path.join(args.save_root, 'workload-{}.csv'.format(repeat_id)))
                print(workload.groupby(["application", "num_gpus", "target_batch_size"])
                    .size().reset_index(name="count"))