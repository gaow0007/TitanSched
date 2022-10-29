import os, sys 
import json
import numpy as np 

NLP_Models = ['roberta-base', 'roberta-large']
NLP_Datasets = ['wnli', 'rte', 'mrpc', 'stsb', 'sst2', 'qnli', 'qqp',  'mnli', "ag_news", "wikitext-103", "snli",]
NLP_prefix = 'glue'
CV_Models = ['vit', 'vit-large']
CV_Datasets = ['Bingsu/Cat_and_Dog', 'frgfm/imagenette', 'cifar100',  'fashion_mnist',  'mnist', 'food101', 'imagenet-split-0', 'imagenet-split-1', 'imagenet-split-2']
CV_prefix = 'image_classification'

MAX_EPOCH=10
WIKI_DATASETS = ['wikitext-103', 'wikitext-2']


# task_scale = {'wnli' : 635,
#     'rte' : 2490,
#     'mrpc' : 3668,
#     'stsb' : 5749,
#     'sst2' : 67349,
#     'ag_news' : 120000,
#     'qnli' : 104743,
#     'qqp' : 363846,
#     'mnli' : 392702,
#     'wikitext-103':1801350,
# }

TaskScale = {
    'wnli' : 159,
    'rte' : 623,
    'mrpc' : 917,
    'stsb' : 1438,
    'sst2' : 16838,
    'qnli' : 26186,
    'qqp' : 90962,
    'mnli' : 98176,
    'snli' : 137344,
    'ag_news' : 30000,
    'wikitext-103':59061,
}
# ['wnli', 'rte', 'mrpc', 'stsb',  'qnli',  'mnli', 'ag_news']
BestHPO = {
    "roberta-base": {
        'rte'  : ('1e-4', 1),
        'mrpc' : ('1e-3', 8),
        'stsb' : ('1e-4', 1),
        'sst2' : None,
        'qnli' :('4e-5', 8),
        'qqp'  : ('1e-4', 4),
        'mnli' : None,
        'snli' : ('1e-4', 4),
        'ag_news' : ('1e-4', 1),
        'wikitext-103': None,
    },
    "roberta-large": {
        'rte'  : None,
        'mrpc' : ('1e-4', 4),
        'stsb' : ('4e-4', 12),
        'sst2' : None,
        'qnli' : ('1e-4', 8),
        'qqp'  : ('4e-5', 8),
        'mnli' : ('4e-5', 12),
        'snli' : ('4e-5', 12),
        'ag_news' : None,
        'wikitext-103': None,
    },
}

def metrick_key_generator(dataset, prefix=''): 
    if dataset in ['mrpc', 'qqp']: 
        metric_key = 'eval_f1'
    elif dataset in 'stsb': 
        metric_key = 'eval_pearson'
    elif dataset in WIKI_DATASETS:
        metric_key = 'eval_loss'
    else: 
        metric_key = 'eval_accuracy'
    return metric_key if len(prefix) == 0 else '{}_{}'.format(prefix, metric_key)
    

def grab_task_info(path, key_list): 
    task_info = dict() 
    for metric_key in key_list: 
        task_info[metric_key] = list() 
        task_info['{}_epoch'.format(metric_key)] = list() 
    if os.path.exists(path): 
        for date in os.listdir(path): 
            date_path = os.path.join(path, date)
            if not os.path.exists(date_path):
                break 
            for seed_dir in os.listdir(date_path): 
                # load eval accuracy 
                json_name = os.path.join(date_path, seed_dir, 'trainer_state.json')
                if not os.path.exists(json_name): 
                    print('rm -rf {}'.format(date_path))
                    continue 
                with open(json_name, 'r') as f: 
                    data = json.load(f)
                for log_info in data['log_history']: 
                    for metric_key in key_list: 
                        if metric_key in log_info: 
                            if 'loss' in metric_key: 
                                task_info[metric_key].append(-log_info[metric_key])
                            else: 
                                task_info[metric_key].append(log_info[metric_key])
                            # import pdb; pdb.set_trace() 
                            task_info['{}_epoch'.format(metric_key)].append(log_info['epoch'])
    
    return task_info 


def create_if_missing(dict_info, lr, gradient_steps, value): 
    if lr not in dict_info: 
        dict_info[lr] = dict() 
    if gradient_steps not in dict_info[lr]: 
        dict_info[lr][gradient_steps] = dict()
    dict_info[lr][gradient_steps] = value 


def apply_transform(task_name, identifier): 
    return task_name.replace('/', identifier)


def query_index(dataset): 
    all_datasets = NLP_Datasets if dataset in NLP_Datasets else CV_Datasets
    identifier = '#'
    return all_datasets.index(dataset.replace(identifier, '/'))


class TransferTaskingInfo(): 
    def __init__(self, ): 
        self.performance_report = dict() 
        self.build_performance_report(NLP_Models, NLP_Datasets, NLP_prefix, root='checkpoints/glue_transfer')
        self.build_performance_report(CV_Models, CV_Datasets, CV_prefix, root='checkpoints/image_classification_transfer')


class HPOTaskingInfo(): 
    def __init__(self, load_cache=False): 
        self.performance_report = dict() 
        if not os.path.exists('common/cache/hpo_task_info.npy') or not load_cache: 
            self.build_performance_report(NLP_Models, NLP_Datasets, NLP_prefix, root='checkpoints/glue/')
            self.build_performance_report(CV_Models, CV_Datasets, CV_prefix, root='checkpoints/image_classification')
            with open('common/cache/hpo_task_info.npy', 'wb') as f: 
                np.save(f, self.performance_report)
        else: 
            with open('common/cache/hpo_task_info.npy', 'rb') as f: 
                self.performance_report = np.load(f, allow_pickle=True).tolist() 
        self.lr_list = ['1e-5', '2e-5', '4e-5', '1e-4', '4e-4', '1e-3', '1e-2']
        self.gradient_steps_list = [1, 4, 8, 12]
    
    def build_performance_report(self, models, datasets, prefix, root): 
        identifier = '#'
        prefix = 'HPO_'
        for model in models: 
            if model not in self.performance_report: 
                self.performance_report[model] = dict() 
            for dataset in datasets: 
                dataset = apply_transform(dataset, identifier=identifier)
                if dataset not in self.performance_report[model]: 
                    self.performance_report[model][dataset] = dict() 
                for lr in ['1e-5', '2e-5', '4e-5', '1e-4', '4e-4', '1e-3', '1e-2']: 
                    for gradient_steps in [1, 4, 8, 12]: 
                        task_path = os.path.join(root, prefix + '{}.{}.{}.{}'.format(model, dataset, lr, gradient_steps))
                        key_list = [metrick_key_generator(dataset, prefix=''), 'log_history']
                        task_info = grab_task_info(task_path, key_list)
                        create_if_missing(self.performance_report[model][dataset], lr, gradient_steps, task_info)
    
    def query_metric_with_hpo(self, model, dataset, metric_key): 
        identifier = '#'
        dataset = apply_transform(dataset, identifier=identifier)
        task_info = self.performance_report[model][dataset]
        max_lr, max_gradient_steps, max_metric = -1, -1, -1
        for lr in self.lr_list: 
            for gradient_steps in self.gradient_steps_list: 
                if len(task_info[lr][gradient_steps][metric_key]) == 0: 
                    print('please re-execute model {}, dataset {}, lr {}, gradient_steps {}'.format(model, dataset, lr, gradient_steps))
                    continue 
                if max(task_info[lr][gradient_steps][metric_key]) > max_metric: 
                    max_metric = max(task_info[lr][gradient_steps][metric_key])
                    max_gradient_steps = gradient_steps
                    max_lr = lr 

        return max_lr, max_gradient_steps, max_metric

    def query_epoch(self, model, dataset, lr, gradient_steps, metric_key, target_metric): 
        identifier = '#'
        dataset = apply_transform(dataset, identifier=identifier)
        metric_info = self.performance_report[model][dataset][lr][gradient_steps][metric_key]
        epoch_info = self.performance_report[model][dataset][lr][gradient_steps][metric_key + '_epoch']
        min_epoch = None 
        for epoch, metric in zip(epoch_info, metric_info): 
            if target_metric <= metric: 
                if min_epoch is None: 
                    min_epoch = epoch 
                else: 
                    min_epoch = min(epoch, min_epoch)

        return MAX_EPOCH if min_epoch is None else min_epoch

    def query_appropriate_hpo(self, model, dataset, metric_key=None, target_metric=None): 
        assert metric_key is not None and target_metric is not None 
        identifier = '#'
        dataset = apply_transform(dataset, identifier)
        task_info = self.performance_report[model][dataset]
        metric_list = list() 
        for lr in self.lr_list: 
            for gradient_steps in self.gradient_steps_list: 
                # print(model, dataset, lr, gradient_steps, task_info[lr][gradient_steps])
                if len(task_info[lr][gradient_steps][metric_key]) == 0: 
                    continue 
                if max(task_info[lr][gradient_steps][metric_key]) >= target_metric: 
                    # print(task_info[lr][gradient_steps][metric_key + '_epoch'], lr, gradient_steps)
                    # print(task_info[lr][gradient_steps][metric_key], lr, gradient_steps)
                    metric_list.append((max(task_info[lr][gradient_steps][metric_key]), lr, gradient_steps))
        return metric_list


    def query_best_hpo(self, model, dataset, metric_key=None, last_epoch=False): 
        identifier = '#'
        dataset = apply_transform(dataset, identifier)
        task_info = self.performance_report[model][dataset]
        if metric_key is None: 
            metric_key = self.standarized_metric(dataset)

        metric_list = list() 
        if model in BestHPO and dataset in BestHPO[model] and BestHPO[model][dataset] is not None: 
            lr, gradient_steps = BestHPO[model][dataset]
            metric = max(task_info[lr][gradient_steps][metric_key])
            if last_epoch: 
                return task_info[lr][gradient_steps][metric_key][-1], lr, gradient_steps
            else: 
                return (metric, lr, gradient_steps)
            
        for lr in self.lr_list: 
            for gradient_steps in self.gradient_steps_list: 
                # print(model, dataset, lr, gradient_steps, task_info[lr][gradient_steps])
                if len(task_info[lr][gradient_steps][metric_key]) == 0: 
                    metric = -sys.maxsize
                else: 
                    metric =  task_info[lr][gradient_steps][metric_key][-1] if last_epoch else max(task_info[lr][gradient_steps][metric_key])
                    # print(task_info[lr][gradient_steps][metric_key], dataset, metric)
                metric_list.append((metric, lr, gradient_steps))
        metric_list.sort(key=lambda x: -x[0])
        # print(metric_list[0])
        return metric_list[0]

    def query_metric(self, model, dataset, lr=None, gradient_steps=None, metric_key=None, func=None): 
        identifier = '#'
        dataset = apply_transform(dataset, identifier)
        task_info = self.performance_report[model][dataset]
        lr_list = self.lr_list if lr is None else [lr]
        gradient_steps_list = self.gradient_steps_list if gradient_steps is None else [gradient_steps]
        # assert metric_key is not None, 'please provide the concrete `metric_key`'
        metric_list = list() 
        for lr in lr_list: 
            for gradient_steps in gradient_steps_list: 
                if metric_key is not None: 
                    metric_list.append(task_info[lr][gradient_steps][metric_key])
                else: 
                    metric_list.append(task_info[lr][gradient_steps])
        if func is None: 
            return metric_list
        else: 
            return func(metric_list)
    
    def standarized_metric(self, dataset): 
        identifier = '#'
        dataset = apply_transform(dataset, identifier)
        if dataset in ['mrpc', 'qqp']: 
            metric_key = 'eval_f1'
        elif dataset in 'stsb': 
            metric_key = 'eval_pearson'
        elif 'wiki' in dataset: 
            metric_key = 'eval_loss'
        else: 
            metric_key = 'eval_accuracy'
        return metric_key 


class TransferTaskingInfo(): 
    def __init__(self, load_cache=False):
        self.performance_report = dict() 
        if not os.path.exists('common/cache/transfer_task_info.npy') or not load_cache:
            self.build_performance_report(NLP_Models, NLP_Datasets, NLP_prefix, root='checkpoints/glue_transfer/')
            self.build_performance_report(CV_Models, CV_Datasets, CV_prefix, root='checkpoints/image_classification_transfer/')
            with open('common/cache/transfer_task_info.npy', 'wb') as f: 
                np.save(f, self.performance_report)
        else: 
            with open('common/cache/transfer_task_info.npy', 'rb') as f: 
                self.performance_report = np.load(f, allow_pickle=True).tolist() 

    def build_performance_report(self, models, datasets, prefix, root):
        identifier = '#' 
        for model in models: 
            if model not in self.performance_report: 
                self.performance_report[model] = dict() 

            for i in range(len(datasets)): 
                for j in  range(len(datasets)):
                    if i == j: continue 
                    datasetA = apply_transform(datasets[i], identifier)
                    datasetB = apply_transform(datasets[j], identifier)
                    index_key = '_{}.{}.{}.'.format(datasetA, model, datasetB)
                    insert_key = '{},{}'.format(datasetA, datasetB)
                    # if insert_key == 'rte,wnli': 
                    #     import pdb; pdb.set_trace() 
                    task_path_list = [os.path.join(root, task_path) for task_path in os.listdir(root) if index_key in task_path]
                    key_list = [metrick_key_generator(datasetB, prefix=''), 'log_history']
                    rep_key = key_list[0]
                    for task_path in task_path_list:
                        task_info = grab_task_info(task_path, key_list)
                        if len(task_info[rep_key]) > 0: 
                            self.performance_report[model][insert_key] = task_info 
    

    def query_all_metric(self, model, datasetA, datasetB, metric_key): 
        identifier = '#'
        datasetA = apply_transform(datasetA, identifier)
        datasetB = apply_transform(datasetB, identifier)
        insert_key = '{},{}'.format(datasetA, datasetB)
        try: 
            return (self.performance_report[model][insert_key][metric_key])
        except: 
            return None 

    def generate_dataset_key(self, datasetA, datasetB): 
        all_datasets = NLP_Datasets if datasetA in NLP_Datasets else CV_Datasets 
        # assert all_datasets.index(datasetA) < all_datasets.index(datasetB), 'be careful about the passing arguments order'
        identifier = '#'
        datasetA = apply_transform(datasetA, identifier)
        datasetB = apply_transform(datasetB, identifier)
        insert_key = '{},{}'.format(datasetA, datasetB) 
        return insert_key 

    def query_metric(self, model, datasetA, datasetB, metric_key): 
        identifier = '#'
        datasetA = apply_transform(datasetA, identifier)
        datasetB = apply_transform(datasetB, identifier)
        insert_key = '{},{}'.format(datasetA, datasetB)
        
        try: 
            if 'loss' in metric_key: 
                return min(self.performance_report[model][insert_key][metric_key])
            else: 
                return max(self.performance_report[model][insert_key][metric_key])
        except: 
            return None 
    
    def query_epoch(self, model, dataset, insert_key, metric_key, target_metric): 
        # dataset optimized with transfer learning
        identifier = '#'
        dataset = apply_transform(dataset, identifier)
        if insert_key not in self.performance_report[model]: 
            return MAX_EPOCH
        metric_info = self.performance_report[model][insert_key][metric_key]
        epoch_info = self.performance_report[model][insert_key][metric_key + '_epoch']
        min_epoch = None 
        for epoch, metric in zip(epoch_info, metric_info): 
            if target_metric <= metric: 
                if min_epoch is None: 
                    min_epoch = epoch 
                else: 
                    min_epoch = min(epoch, min_epoch)

        return MAX_EPOCH if min_epoch is None else min_epoch
    

    def standarized_metric(self, dataset): 
        identifier = '#'
        dataset = apply_transform(dataset, identifier)
        if dataset in ['mrpc', 'qqp']: 
            metric_key = 'eval_f1'
        elif dataset in 'stsb': 
            metric_key = 'eval_pearson'
        elif 'wiki' in dataset: 
            metric_key = 'eval_loss'
        else: 
            metric_key = 'eval_accuracy'
        return metric_key 


class MultiTaskingInfo(): 
    def __init__(self, load_cache=False):
        self.performance_report = dict() 
        if not os.path.exists('common/cache/multi_task_info.npy') or not load_cache:
            self.build_performance_report(NLP_Models, NLP_Datasets, NLP_prefix, root='checkpoints/glue_multi_tasking/')
            self.build_performance_report(CV_Models, CV_Datasets, CV_prefix, root='checkpoints/image_classification_multi_tasking/')
            with open('common/cache/multi_task_info.npy', 'wb') as f: 
                np.save(f, self.performance_report)
        else: 
            with open('common/cache/multi_task_info.npy', 'rb') as f: 
                self.performance_report = np.load(f, allow_pickle=True).tolist() 

    def build_performance_report(self, models, datasets, prefix, root):
        identifier = '#' 
        for model in models: 
            if model not in self.performance_report: 
                self.performance_report[model] = dict() 

            for i in range(len(datasets)): 
                for j in  range(i + 1, len(datasets)):
                    datasetA = apply_transform(datasets[i], identifier)
                    datasetB = apply_transform(datasets[j], identifier)
                    index_key = '.{},{}.'.format(datasetA, datasetB)
                    insert_key = '{},{}'.format(datasetA, datasetB)
                    if not os.path.exists(root): continue
                    task_path_list = [os.path.join(root, task_path) for task_path in os.listdir(root) if index_key in task_path and model in task_path]
                    key_list = [metrick_key_generator(datasetA, prefix=datasetA), metrick_key_generator(datasetB, prefix=datasetB), 'log_history']
                    rep_key = key_list[0]
                    for task_path in task_path_list: 
                        # print("task_path {}".format(task_path))
                        task_info = grab_task_info(task_path, key_list)
                        if len(task_info[rep_key]) > 0: 
                            self.performance_report[model][insert_key] = task_info 
    
    def generate_dataset_key(self, datasetA, datasetB): 
        all_datasets = NLP_Datasets if datasetA in NLP_Datasets else CV_Datasets 
        # assert all_datasets.index(datasetA) < all_datasets.index(datasetB), 'be careful about the passing arguments order'
        identifier = '#'
        datasetA = apply_transform(datasetA, identifier)
        datasetB = apply_transform(datasetB, identifier)
        if all_datasets.index(datasetA.replace(identifier, '/')) < all_datasets.index(datasetB.replace(identifier, '/')): 
            insert_key = '{},{}'.format(datasetA, datasetB)
        else: 
            insert_key = '{},{}'.format(datasetB, datasetA)
        return insert_key 
    
    def query_metric(self, model, datasetA, datasetB, metric_key): 
        all_datasets = NLP_Datasets if datasetA in NLP_Datasets else CV_Datasets 
        # assert all_datasets.index(datasetA) < all_datasets.index(datasetB), 'be careful about the passing arguments order'
        identifier = '#'
        datasetA = apply_transform(datasetA, identifier)
        datasetB = apply_transform(datasetB, identifier)
        if all_datasets.index(datasetA.replace(identifier, '/')) < all_datasets.index(datasetB.replace(identifier, '/')): 
            insert_key = '{},{}'.format(datasetA, datasetB)
        else: 
            insert_key = '{},{}'.format(datasetB, datasetA)
        
        try: 
            if 'loss' in metric_key: 
                return min(self.performance_report[model][insert_key][metric_key])
            else: 
                return max(self.performance_report[model][insert_key][metric_key])
        except: 
            return None 

    def query_epoch(self, model, datasetA, datasetB, metric_key, target_metric): 
        all_datasets = NLP_Datasets if datasetA in NLP_Datasets else CV_Datasets 
        # assert all_datasets.index(datasetA) < all_datasets.index(datasetB), 'be careful about the passing arguments order'
        identifier = '#'
        datasetA = apply_transform(datasetA, identifier)
        datasetB = apply_transform(datasetB, identifier)
        if all_datasets.index(datasetA.replace(identifier, '/')) < all_datasets.index(datasetB.replace(identifier, '/')): 
            insert_key = '{},{}'.format(datasetA, datasetB)
        else: 
            insert_key = '{},{}'.format(datasetB, datasetA)
        
        if insert_key not in self.performance_report[model]: 
            return MAX_EPOCH

        metric_key = apply_transform(metric_key, identifier)
        metric_info = self.performance_report[model][insert_key][metric_key]
        epoch_info = self.performance_report[model][insert_key][metric_key + '_epoch']
        min_epoch = None 
        for epoch, metric in zip(epoch_info, metric_info): 
            if target_metric <= metric: 
                if min_epoch is None: 
                    min_epoch = epoch 
                else: 
                    min_epoch = min(epoch, min_epoch)

        return MAX_EPOCH if min_epoch is None else min_epoch
    