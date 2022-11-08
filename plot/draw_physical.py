import os, sys
import matplotlib.pyplot as plt 
import seaborn
import matplotlib
import numpy as np 
import math 
import matplotlib
import pandas as pd
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
    'fontsize': 18 + 4+4, 
    'linewidth': 6, 
    'scatter_markersize': 400, 
    'line_markersize': 20, 
    'width': 2, 
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
    if isinstance(ncols, tuple): 
        fig, axes = matplotlib.pyplot.subplots(ncols[0], ncols[1])
        fig.set_size_inches(w=ncols[1]* 4*3, h=3*ncols[0])
        axes = [axes[0], axes[1]]
        # axes = [axes[j][i] for i in range(ncols[0]) for j in range(ncols[1])]
        # import pdb; pdb.set_trace() 
    else: 
        fig, axes = matplotlib.pyplot.subplots(1, ncols)
        if ncols == 1: 
            axes = [axes]
        fig.set_size_inches(w=ncols* 4, h=3)

    for ax in axes: 
        apply_grid(ax, **kwargs)
        apply_spine(ax, **kwargs)

    return fig, axes 


def plot_bar_by_method(ax, info_by_method, **kwargs): 
    apply_log(ax, **kwargs)
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

        if len(norm_list) > 0: 
            print(ident)
            print(norm_list, len(norm_list))
            print(value_list, len(value_list))
        if kwargs.get('norm'): 
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
        if kwargs.get('disable_legend'): 
            rect = ax.bar(x_list, value_list, yerr=error_list, width=width, color=color_list[idx], hatch=hatch_list[idx], alpha=0.75, edgecolor='black', capsize=0)
        else: 
            rect = ax.bar(x_list, value_list, yerr=error_list, width=width, color=color_list[idx], hatch=hatch_list[idx], alpha=0.75, edgecolor='black', capsize=0, label=ident)

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
        ax.set_xticks([])


def main_jct_bar(info_by_method, save_path, ax, fig, **new_template):
    # for ax in axes: 
    if True: 
        x_list = [i for i in range(10)]
        y_list = [np.random.randn(1) for x in x_list]
        template.update(new_template)


        plot_bar_by_method(ax, info_by_method, **template)
        ax.set_ylabel(' Norm. \n JCT', fontsize=template['fontsize'])
        ax.set_yticks([1, 2, 4, 8])
        ax.tick_params(axis='both', labelsize=template['fontsize'])
        ax.set_ylim(0, 10)

        if template.get('xname', None) is not None: 
            ax.set_xlabel(template.get('xname'), fontsize=template['fontsize'])

        # import pdb; pdb.set_trace() 
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # axes_list = [1.2, 3.2, 5.2, 7.2]
        # ax.set_xticks([val - 0.25 for val in axes_list])
        # ax.set_xticklabels(['0.5', '1.0', '1.5', '2.0'], fontsize=template['fontsize'], rotation=0)
        # ax.set_xlabel('Relative Workload Density', fontsize=template['fontsize'])
    # plt.legend() 


def main_makespan_bar(info_by_method, save_path, ax, fig, **new_template):
    # for ax in axes: 
    if True: 
        x_list = [i for i in range(10)]
        y_list = [np.random.randn(1) for x in x_list]
        template.update(new_template)


        plot_bar_by_method(ax, info_by_method, **template)
        ax.set_ylabel('  Norm.  \n Tail JCT', fontsize=template['fontsize'])
        ax.tick_params(axis='both', labelsize=template['fontsize'])
        ax.set_yticks([1, 2, 4, 8])
        ax.set_ylim(0, 10)
        

        if template.get('xname', None) is not None: 
            ax.set_xlabel(template.get('xname'), fontsize=template['fontsize'])


if __name__ == '__main__': 
    import glob 
    model_name_list = ['roberta-base', 'roberta-large', 'vit', 'vit-large']
    
    fig, axes = init_plot((2, 1), grid=True)

    jct_info_by_method = list() # ('titan', '-True'), 
    makespan_info_by_method = list() 
    fft_info_by_method = list() 
    for (schedule, sched_verbose) in [('titan', '-False-True'), ('pollux', ''),  ('optimus', ''), ('tiresias', '')]:
        jct_list = list() 
        makespan_list = list() 
        for model_id, model_name in enumerate(model_name_list):
            for trace_path in sorted(glob.glob('trace/FM-*')): 
                if '320' not in trace_path: 
                    continue 
                if os.path.isdir(trace_path) and trace_path.endswith(model_name): 
                    trace_ident = os.path.basename(trace_path)
                    # if schedule == 'titan': 
                    trace_ident = trace_ident + sched_verbose
                    csv_name = os.path.join('result/', schedule, trace_ident, '{}.csv'.format(schedule))
                    df = pd.read_csv(csv_name)
                    num_job = 1.0 * len(df)
                    jct = 0
                    min_time = sys.maxsize
                    max_time = 0
                    makespan = 0 
                    tot_jct_list = list() 
                    for idx, job in df.iterrows(): 
                        jct += (job.completion_time - job.submission_time) / num_job
                        tot_jct_list.append(job.completion_time - job.submission_time)
                        min_time = min(job.submission_time, min_time)
                        max_time = max(job.completion_time, max_time)
                    jct_list.append(jct / 3600)
                    makespan_list.append(max(tot_jct_list))
        sched_verbose = ''
        jct_info_by_method.append([schedule + sched_verbose, jct_list, [0 for jct in jct_list]])
        makespan_info_by_method.append([schedule + sched_verbose, makespan_list, [0 for makespan in makespan_list]])

    
    new_template =  {
        "norm": False, 
        "width": 0.3, 
        "autolabel": False, 
        'norm': True,
        'logy': 2,
        'yname': True if model_id == 0 else None, 
        'disable_legend': model_id>0,
        'xname': None,
    }
    
    main_jct_bar(info_by_method=jct_info_by_method, save_path='plot/images/{}.jpg'.format(model_name), ax=axes[0], fig=fig,  **new_template)

    new_template =  {
        "norm": False, 
        "width": 0.3, 
        "autolabel": False, 
        'norm': True,
        'logy': 2,
        'disable_legend': True,
        'yname': True if model_id == 0 else None, 
        'xname': model_name,
    }

    main_makespan_bar(info_by_method=makespan_info_by_method, save_path='plot/images/{}.jpg'.format(model_name), ax=axes[1], fig=fig,  **new_template)

    save_path = 'plot/images/physical.jpg'
    
    fig.legend(fontsize=template['fontsize'], loc='upper center', ncol=4, bbox_to_anchor=(0.55, 1.05), fancybox=True, shadow=False) 

    plt.savefig(save_path, bbox_inches='tight')
    plt.close() 


        # info_by_method = [
        #     ['Tiresias',[1.75, 1.67, 2.87, 4.57], [0, 0, 0, 0] ],
        #     ['SRTF ', [1.74, 1.68, 1.99, 3.02], [0, 0, 0, 0] ], 
        #     ['Titan', [0.89, 1.04, 1.44, 1.88], [0, 0, 0, 0]]
        # ]
                    

