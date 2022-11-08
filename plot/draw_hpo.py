import os, sys
import matplotlib.pyplot as plt 
import seaborn
import matplotlib
import numpy as np 
import math 
import matplotlib
import pandas as pd
import json 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


color_list = ['tab:orange',
            'tab:blue',
            'tab:green',
            'tab:red',
            'tab:purple',
            'tab:brown',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'tab:cyan']

hatch_list = [
    '', 
    '/', 
    '\\'
    '///', 
    '--', 
    '+', 
    'x'
    '*', 
    'o', 
    'O', 
    '.'
]

line_style_list = [
    '-', 
    '--', 
    '-.', 

]

marker_list = [
    '',
    'o', 
    'v',
    '^', 
    'X', 
    'D'
    's', 
]

template = {
    'fontsize': 18, 
    'linewidth': 6, 
    'scatter_markersize': 400, 
    'line_markersize': 20, 
    'width': 0.3, 
}

def autolabel_percent(rects, ax, value_list, error_list=None, str_func=None):
    if str_func is None: 
        str_func = lambda x: '%.2f'%(x)

    if error_list is None: 
        error_list = [0 for _ in value_list]

    for idx, rect in enumerate(rects):
        if value_list[idx] is None: continue
        height = rect.get_height()
        ax.annotate(str_func(value_list[idx]),
                    xy=(rect.get_x() + rect.get_width() / 2, height+error_list[idx]),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16, fontweight='bold')


def check_before_run(**kwargs): 
    if   kwargs['full'] + kwargs['half'] + kwargs['forth'] > 1: 
        return False 
    return True 


def apply_grid(ax, **kwargs): 
    if kwargs.get('grid'): 
        if not (kwargs.get('ygrid') or kwargs.get('xgrid')): 
            ax.grid(linestyle='-.', linewidth=1, alpha=0.5)

    if kwargs.get('ygrid'): 
        ax.grid(linestyle='-.', linewidth=1, alpha=0.5, axis='y')
    if kwargs.get('xgrid'): 
        ax.grid(linestyle='-.', linewidth=1, alpha=0.5, axis='x')


def apply_spine(ax, **kwargs): 
    if kwargs.get('spines'): 
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')


def apply_font(kwargs): 
    font = {'family' : 'serif',
            'size'   : 18}
    if kwargs.get('font'): 
        font.update(kwargs.get('font'))
    matplotlib.rc('font', **font)


def apply_log(ax, **kwargs): 
    if kwargs.get('logx'): 
        ax.set_xscale('log', basex=kwargs.get('logx'))
    if kwargs.get('logy') > 0: 
        ax.set_yscale('log', basey=kwargs.get('logy'))

def init_plot(ncols, **kwargs): 
    # if len(kwargs) > 0: 
    #     assert check_before_run(kwargs)
    
    apply_font(kwargs)
    fig, axes = matplotlib.pyplot.subplots(1, ncols)
    if ncols == 1: 
        axes = [axes]
    fig.set_size_inches(w=ncols* 6, h=3)

    for ax in axes: 
        apply_grid(ax, **kwargs)
        apply_spine(ax, **kwargs)
        apply_log(ax, **kwargs)

    return fig, axes 


def plot_bar_by_method(ax, info_by_method, **kwargs): 
    width = kwargs.get('width', 0.3)
    interval = int(math.ceil(width * len(info_by_method)) * 2)
    if kwargs.get('norm'): 
        norm_list = list() 
    
    for idx, (ident, y_list, error_list) in enumerate(info_by_method): 
        x_list = list() 
        base = width * ( (len(info_by_method) - 1) // 2 + 0.5 * (len(info_by_method) - 1) % 2 ) + idx * width
        value_list = list() 
        for idy, y in enumerate(y_list): 
            x_list.append(base + idy * interval)
            value_list.append(y)


        if kwargs.get('norm'): 
            if len(norm_list) > 0: 
                print(ident)
                print(norm_list, len(norm_list))
                print(value_list, len(value_list))

            if len(norm_list) == 0: 
                norm_list = [val for val in value_list]
                value_list = [1. for _ in value_list]
            else: 
                value_list = [val / norm for val, norm in zip(value_list, norm_list)]
                # import pdb; pdb.set_trace() 
        # import pdb; pdb.set_trace() 
        if error_list is None: 
            error_list = [0 for _ in y_list]
        
        def cap(value): 
            return value 
            # if value < 5: 
            #     return value 
            # else: 
            #     return 5 + (value - 5) / 5
        value_list = [cap(value) for value in  value_list]
        
        if kwargs.get('barh', False) == True: 
            rect = ax.barh(x_list, value_list, xerr=error_list, height=width, color=color_list[idx], hatch=hatch_list[0], alpha=0.75, edgecolor='black', capsize=5, label=ident)
        else:
            rect = ax.bar(x_list, value_list, yerr=error_list, width=width, color=color_list[idx], hatch=hatch_list[0], alpha=0.75, edgecolor='black', capsize=5, label=ident) 

        print('x_list', x_list)    
        print('y_list', y_list)
        if kwargs.get('autolabel'): 
            #autolabel_percent(rects, ax, value_list, error_list=None, str_func=None):
            str_func = None 
            if kwargs.get('norm'): 
                str_func = lambda x: '%.2f'%(x)
            elif 'int' in str(type(y_list[0])):
                str_func = lambda x: '%d'%(x)
            print('value_list {}'.format(value_list))
            # autolabel_percent(rect, ax, value_list, error_list=error_list, str_func=str_func)



def main(info_by_method, save_path): 

    fig, axes = init_plot(1, grid=True, logy=template.get('logy', 0))

    for ax in axes: 
        x_list = [i for i in range(10)]
        y_list = [np.random.randn(1) for x in x_list]
        # ax.plot(x_list, y_list, linestyle=line_style_list[1], linewidth=template['linewidth'], color=color_list[0], \
        #     marker=marker_list[3], markersize=template['line_markersize'], markerfacecolor=color_list[1], markeredgecolor=color_list[1])
        # ax.scatter(x_list, y_list, marker=marker_list[2], s=template['scatter_markersize'])
        # info_by_method = [
        #     ['Tiresias',[1.75, 1.67, 2.87, 4.57], [0, 0, 0, 0] ],
        #     ['SRTF ', [1.74, 1.68, 1.99, 3.02], [0, 0, 0, 0] ], 
        #     ['Titan', [0.89, 1.04, 1.44, 1.88], [0, 0, 0, 0]]
        # ]




        plot_bar_by_method(ax, info_by_method, **template)
        # ax.set_xlim(-0.5, 4)
        # ax.set_ylim(0, 1, 2)
        ax.set_ylabel('Avg. JCT (Hour)', fontsize=template['fontsize'])
        # import pdb; pdb.set_trace() 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # axes_list = [1.2, 3.2, 5.2, 7.2]
        if False: 
            axes_list = [1, 5, 9, 13]
            ax.set_xticks([val + 0.5 for val in axes_list])
            ax.set_xticklabels(['0.5', '1.0', '1.5', '2.0'], fontsize=template['fontsize'], rotation=0)
        else: 
            axes_list = [1, 3, 5]
            ax.set_xticks([val + 1.5 for val in axes_list])
            ax.set_xticklabels(['1.0', '1.5', '2.0'], fontsize=template['fontsize'], rotation=0)
        ax.set_xlabel('Relative Workload Density', fontsize=template['fontsize'])
    # plt.legend() 
    # fig.legend(fontsize=template['fontsize'], loc='upper center', ncol=3)
    fig.legend(fontsize=template['fontsize'], loc='upper center', ncol=3, bbox_to_anchor=(0.55, 1.05), fancybox=True, shadow=False) 

    plt.savefig(save_path, bbox_inches='tight') 
    plt.close() 

def cal_best_metric(best_trial): 
    best_metric = 0 
    for trial in best_trial: 
        
        best_metric = max(best_metric, trial[-1])
    return best_metric 

def cal_metric_difference(hpo_results, application_list):
    metric_difference_list = list() 
    for application in application_list: 
        true_best_metric = cal_best_metric(hpo_results['True'][application]['best_trials'])
        false_best_metric = cal_best_metric(hpo_results['False'][application]['best_trials'])
        ratio = (false_best_metric - true_best_metric) / true_best_metric
        if ratio != 0: 
            metric_difference_list.append(ratio)
    # print(metric_difference_list)
    # assert len(metric_difference_list) <= 1
    return sum(metric_difference_list)


def cal_jct_difference(hpo_results, application_list):
    jct_info = {
        'True': 0, 
        'False': 0,
    }
    for application in application_list: 
        best_trials = hpo_results['True'][application]['best_trials']
        true_jct = hpo_results['True'][application]['jct']
        false_jct = hpo_results['False'][application]['jct']
        jct_info['True'] += true_jct 
        jct_info['False'] += false_jct 
    return  jct_info['False'] / jct_info['True'] - 1


if __name__ == '__main__': 
    import glob 
    model_name_list = ['roberta-base',  'roberta-large', 'vit', 'vit-large']
    other_name_list = ['Roberta-base', 'Roberta-Large', 'ViT', 'ViT-Large']
    rows = list() 
    rows.append(['Model', 'Metric Diff.', 'JCT Speedup'])
    log_info = ''
    root = 'result/hpo_titan/'
    for model_name in model_name_list:
        metric_difference_list = list() 
        jct_difference_list = list() 
        schedule = 'titan'
        # for sched_verbose in ['True', 'False']: 
        if True: 
            for trace_path in sorted(glob.glob('trace/HPO/FM-{}/*'.format(model_name))): 
                if True: 
                    trace_ident = os.path.basename(trace_path).split('-')[-1].split('.csv')[0]
                    if int(trace_ident) < 1 or int(trace_ident) > 50: continue 
                    hpo_results = dict() 
                    for sched_verbose in ['False', 'True']: 
                        json_name = os.path.join('result/hpo_titan/', 'FM-{}-{}'.format(model_name, sched_verbose), trace_ident, 'hpo_{}_results.json'.format(schedule))
                        if not os.path.exists(json_name): 
                            continue 
                        with open(json_name, 'r') as f: 
                            results = json.load(f)
                        hpo_results[sched_verbose] = results
                    if len(hpo_results) < 2: 
                        import pdb; pdb.set_trace() 
                    application_list = sorted(hpo_results['True'].keys())
                    metric_diff = cal_metric_difference(hpo_results, application_list)
                    jct_diff = cal_jct_difference(hpo_results, application_list)
                    metric_difference_list.append(metric_diff)
                    jct_difference_list.append(jct_diff)
        print('model name {}, metric {}, jct {}'.format(other_name_list[model_name_list.index(model_name)], np.mean(metric_difference_list), np.mean(jct_difference_list)))
        metric_diff = 100 * np.mean(metric_difference_list)
        jct_diff = np.mean(jct_difference_list) * 100
        rows.append([model_name, '{:.2f}\%'.format(metric_diff), '{:.2f}%'.format(jct_diff)])

# from tabulate import tabulate
from texttable import Texttable

import latextable

table = Texttable()
table.set_cols_align(["c", "c", "c"])
table.set_cols_valign(["m", "m", "m"])
table.add_rows(rows)
table.set_deco(Texttable.HEADER | Texttable.VLINES)

print(table.draw())
print('\nLatextable Output:')
print(latextable.draw_latex(table, caption="HPO Performance Comparison over FMs.", label="tab:hpo-results"))