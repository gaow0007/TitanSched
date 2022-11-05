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
GPU_LIMIT=16
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
    helios_trace = parse_helios('trace/full_trace/Helios.csv')
    bm_trace = parse_BM('trace/full_trace/BM.csv')
    rng = random.Random(seed)
    rng2 = random.Random(seed + 3)
    rng3 = random.Random(seed + 4)
    job_rng = np.random.default_rng(seed)
    from client.application.foundation_model import FMStats

    for trace_name, trace in [('BM', bm_trace)]: 
    # for trace_name, trace in [('Helios', helios_trace)]: 
        sample = trace.loc[(trace.duration >= args.min_time) & (trace.duration < args.max_time)]
        sample.insert(1, 'gpu_time', sample.num_gpus * sample.duration, True)
        if hasattr(args, 'gpu_limit') and args.gpu_limit > 0: 
            sample.num_gpus = sample.num_gpus.apply(lambda gpu: min(gpu, args.gpu_limit))
        else: 
            sample.num_gpus = sample.num_gpus.apply(lambda gpu: min(gpu, GPU_LIMIT))
        
        sample = sample.loc[sample.gpu_time < 100 * 3600]
        if True:
            sample.submission_time = sample.submission_time.apply(lambda st : st % (24 * 60 * 60))
            # sample = sample.loc[sample.submission_time < 16 * 3600]
            # sample = sample.loc[sample.submission_time > 8 * 3600]
            # sample.submission_time = sample.submission_time.apply(lambda st : st - (8 * 60 * 60))
            sample = sample.loc[sample.submission_time < 12 * 3600]
        else:
            # day = 264
            # left = day * 24 * 3600
            # right = left + 5 * 24 * 3600
            # sample = sample.loc[sample.submission_time < right]
            # sample = sample.loc[sample.submission_time > left]
            # sample.submission_time = sample.submission_time.apply(lambda st : st % (8 * 60 * 60))

            for i in range(365): 
                left = i * 24 * 3600 + 8 * 3600
                sample = sample.loc[sample.submission_time > left]
                new_sample = sample.loc[sample.submission_time < left + 8 * 3600]
                if len(new_sample) > 0:
                    print('day {}, sample length {}'.format(i, len(new_sample)))
            import pdb; pdb.set_trace() 
            sample = sample.loc[sample.submission_time < 8 * 3600]
        
        if args.num_jobs > 0: 
            sample = sample.sample(n=args.num_jobs, random_state=rng.randint(0, 1 << 32))
        for key in ['submission_time', 'num_gpus', 'duration']: 
            if hasattr(sample, key): 
                sample = sample.astype({key: 'int32'}) 
        records = list() 
        application_list = [key for key in APPLICATIONS.keys()]
        small_list = [APPLICATIONS[app].name for app in application_list if APPLICATIONS[app].scale == 'small']
        medium_list = [APPLICATIONS[app].name for app in application_list if APPLICATIONS[app].scale == 'medium']
        large_list = [APPLICATIONS[app].name for app in application_list if APPLICATIONS[app].scale == 'large']
        xlarge_list = [APPLICATIONS[app].name for app in application_list if APPLICATIONS[app].scale == 'xlarge']
        if len(xlarge_list) == 0: 
            xlarge_list = large_list

        for row in sample.itertuples(): 
            rec = {"submission_time": row.submission_time}

            num_gpus = row.num_gpus
            rec['application'] = application_list[0]
            if row.gpu_time <= 1 * 3600:
                rec["application"] =  rng.choice(small_list)
            elif row.gpu_time <= 5 * 3600:
                rec["application"] = rng.choice(medium_list)
            else: # row.gpu_time < 100 * 3600:
                rec["application"] = rng.choice(large_list)
                subset = sample[sample.duration <= 24 * 3600]
                subset = subset[subset.gpu_time >= 5 * 3600]
                subset = subset[subset.gpu_time < 10 * 3600]
                num_gpus = rng3.choice(subset.num_gpus.to_list())
                
            if args.add_metric: 
                model, dataet = rec["application"].split('@')
                metric, lr, gradient_steps = FMStats.hpo_info.query_best_hpo(model, dataet, last_epoch=False)
                if 'vit' in model: 
                    target_batch_size = 12 * 4
                elif 'roberta' in model: 
                    target_batch_size = 4 * 4
                rec["target_batch_size"] = target_batch_size * gradient_steps
                rec["target_metric"] = metric
                rec["target_lr"] = str(lr)
                rec["target_gradient_steps"] = gradient_steps
                rec["num_gpus"] = min(num_gpus, rec["target_batch_size"] // APPLICATIONS[rec['application']].min_local_bsz)
                if rec["num_gpus"] == 0: 
                    import pdb; pdb.set_trace() 
            else: 
                rec["target_batch_size"] = APPLICATIONS[rec["application"]].max_batch_size
                rec["num_gpus"] = num_gpus
            
            records.append(rec)

        # sample.drop(columns=['gpu_time'], inplace=True)
        # sample.set_index("name").to_csv('{}/{}.csv'.format(args.save_root, trace_name))
        records.sort(key=lambda v: v["submission_time"])
        for idx, rec in enumerate(records):
            rec["name"] = "{}-{}".format(rec["application"], idx)
    if args.add_metric: 
        colums = ("name", "submission_time", "application", "num_gpus", "target_batch_size", "target_lr", "target_gradient_steps", "target_metric")
    else: 
        colums = ("name", "submission_time", "application", "num_gpus", "target_batch_size")
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
    if args.seed < 0: 
        args.seed = seed
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
                if (model_name + '@') in task: 
                    APPLICATIONS[task] = FOUNDATIONMODELAPPLICATIONS[task]
            for repeat_id in range(args.repeat_number): 

                workload = main(model_name, APPLICATIONS)
                csv = workload.set_index("name").to_csv(os.path.join(args.save_root, 'workload-{}.csv'.format(repeat_id)))
                print(workload.groupby(["application", "num_gpus", "target_batch_size"])
                    .size().reset_index(name="count"))