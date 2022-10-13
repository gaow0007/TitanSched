import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from brokenaxes import brokenaxes

colors = [
    "antiquewhite",
    "aquamarine",
    "azure",
    "beige",
    "bisque3",
    "burlywood",
    "chartreuse1",
    "coral1",
    "crimson",
    "darkgoldenrod",
    "darkgreen",
    "darkorchid2",
    "firebrick1",
    "gold1"
]

LOWEST=-1
ACCEPTED_SLO = 0
UNACCEPTED_SLO = 1
BEST_EFFORT = 2


def autolabel_int(rects, ax):
    return 
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%d'%(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')


def autolabel(rects, ax):
    return 
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.1f'%(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')


def autolabel_percent_int(rects, ax):
    return 
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.1f'%(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')


def filename_to_method(name):
    if 'fifo' in name:
        return r'$Yarn-CS$'
    elif 'yarn' in name:
        return r'$Yarn-CS$'
    if 'aggressive' in name:
        return r'$Chronus-A$'
    elif 'conserative' in name:
        return r'$Chronus-C$'
    elif 'advanced-time-aware-with-lease' in name:
        return r'$Chronus$'
    elif 'time-aware-with-lease' in name:
        return r'$Chronus$'
    elif 'time-aware' in name:
        return r'$EDF$'
    elif 'dlas' in name:
        return r'$Tiresias$'
    elif 'themis' in name:
        return r'$Themis$'
    elif 'srtf' in name:
        return r'$SRTF$'
    elif 'tetri' in name:
        return r'$TetriSched$'
    elif 'genie' in name:
        return r'$GENIE$'
    elif 'sigma' in name:
        return r'$3Sigma$'
    else:
        raise NotImplementedError



def filename_to_color(name):
    if 'fifo' in name:
        return 'brown'
    elif 'yarn' in name:
        return 'brown'
    if 'aggressive' in name:
        return 'cyan'
    elif 'conserative' in name:
        return 'lightblue'
    elif 'time-aware-with-lease' in name:
        return 'dodgerblue'
    elif 'time-aware' in name:
        return 'green'
    elif 'dlas' in name:
        return 'orchid'
    elif 'themis' in name:
        return 'pink'
    elif 'srtf' in name:
        return 'wheat'
    elif 'tetri' in name:
        return 'olive'
    elif 'genie' in name:
        return 'skyblue'
    elif 'sigma' in name:
        return 'brown'
    else:
        raise NotImplementedError


def get_cmap(n, name='Dark2_r'):
    return plt.cm.get_cmap(name, n)


def get_cdf(data):
    sorted_data = sorted(data)
    p = 100. * np.arange(len(sorted_data)) / (len(sorted_data) - 1)
    return sorted_data, p


def get_value_from_filename(name: str) -> str:
    return filename_to_method(name.split('.csv')[0])




def trace2linestyle(trace_file):
    if 'Philly_SLO' in trace_file:
        return '-'
    if 'Philly_MIX' in trace_file:
        return '-.'
    if 'Helios_SLO' in trace_file:
        return '--'
    if 'Helios_MIX' in trace_file:
        return ':'
    if 'sh40' in trace_file:
        return '-'
    if 'philly_trace' in trace_file:
        return '--'

def trace2color(trace_file):
    if 'Philly_SLO' in trace_file:
        return 'green'
    if 'Philly_MIX' in trace_file:
        return 'cyan'
    if 'Helios_SLO' in trace_file:
        return 'red'
    if 'Helios_MIX' in trace_file:
        return 'pink'
    if 'sh40' in trace_file:
        return 'tab:blue'
    if 'philly_trace' in trace_file:
        return 'tab:orange'

def trace2name(trace_file):
    for name in ['Philly_SLO', 'Philly_MIX', 'Helios_SLO', 'Helios_MIX']:
        if name in trace_file:
            return name
    if 'sh40' in trace_file:
        return 'Helios'
    if 'philly_trace' in trace_file:
        return 'Philly'


def plot_job_duration_distribution(trace_file_list, save_path):
    # cmap = get_cmap(10)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    for trace_file in trace_file_list:
        
        df = pd.read_csv(trace_file)
        duration_list = df['duration']
        # duration_list = [duration * 60 for duration in duration_list]
        duration_list =[duration for duration in duration_list] #  if duration > 300]
        x, y = get_cdf(duration_list)
        ax.plot(x, y, linewidth=4, color=trace2color(trace_file), label=trace2name(trace_file)) # , linestyle=trace2linestyle(trace_file))

        x_list = [5]
        y_list = [0]
        index = -1
        for i in range(len(y)):
            if y[i] >= 80:
                index = i
                break
        x_list.append(x[index])
        y_list.append(y[index])
        if False:
            for pointx, pointy in zip(x_list, y_list):
                plt.plot([pointx], [pointy], 'o', color='black')
                plt.annotate('(%d,%d)'%(pointx, int(pointy)), xy=(pointx, pointy), textcoords="data", ha='center', va='bottom', fontsize=4) # , fontweight='bold')
    
    delta=5
    plt.tick_params(axis='y', which='major', labelsize=fontsize+delta)
    plt.tick_params(axis='x', which='major', labelsize=fontsize+delta)
    plt.xscale('log')
    plt.xticks([10**i for i in range(7)], [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$', r'$10^{5}$', r'$10^{6}$'])
    plt.xlabel('Job Duration (s)', fontsize=fontsize + delta)
    plt.ylabel('CDF', fontsize=fontsize + delta)

    plt.xlim(1, 10**6)
    plt.ylim(0, 100)
    plt.legend(fontsize=fontsize+delta)
    plt.grid(axis='both',  linestyle='-.')
    fig.set_size_inches(w=6/0.75, h=6)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def plot_job_ddl_duration_correlation(trace_file, save_path):
    cmap = get_cmap(10)
    df = pd.read_csv(trace_file)
    x = df['duration'].to_list()
    y = df['expect_time'].to_list()
    print(max(x), max(y))
    plt.scatter(x, y, color='#054E9F')
    plt.xlabel(r'$Job Duration$')
    plt.ylabel(r'$DDLTime$')
    plt.title(r'$Duration-DDL Correlation$')
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def plot_job_ddl_distribution(trace_file, save_path):
    cmap = get_cmap(10)
    df = pd.read_csv(trace_file)
    expect_list = df['expect_time']
    x, y = get_cdf(expect_list)
    plt.plot(x, y, color=cmap(2))
    plt.xlabel(r'$DDLTime$')
    plt.ylabel(r'$CDF$')
    plt.title(r'$Deadline CDF$')

    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')



def prepare_draw(input_dir):
    work_load_list = sorted([work_load for work_load in os.listdir(input_dir)])
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), sharex=True)
    axes = axes.flatten()
    return fig, axes, work_load_list

def prepare_strict_draw(input_dir):
    work_load_list = sorted(['Helios_SLO', 'Helios_MIX1', 'Philly_SLO', 'Philly_MIX1'])
    fig, axes = plt.subplots(1, 1, figsize=plt.figaspect(0.5), sharex=True)
    return fig, [axes], work_load_list

def prepare_all_draw(input_dir):
    work_load_list = sorted(['Helios_SLO', 'Helios_MIX1', 'Helios_MIX2']) # , 'Philly_SLO', 'Philly_MIX1', 'Philly_MIX2'])
    # work_load_list = ['Helios_MIX1', 'Helios_MIX2', 'Helios_SLO', 'Philly_MIX1', 'Philly_MIX2']
    fig, axes = plt.subplots(1, 1, figsize=plt.figaspect(0.5), sharex=True)
    return fig, [axes], work_load_list

def prepare_weighted_draw(input_dir):
    work_load_list = sorted(['Helios_MIX2', 'Philly_MIX2'])
    # work_load_list = [ 'Helios_MIX2']
    fig, axes = plt.subplots(1, 1, figsize=plt.figaspect(0.5), sharex=True)
    return fig, [axes], work_load_list

def prepare_single_draw(input_dir):
    work_load_list = sorted(['Helios_SLO', 'Helios_MIX1', 'Helios_MIX2', 'Philly_SLO', 'Philly_MIX1', 'Philly_MIX2'])
    # work_load_list = sorted([work_load for work_load in os.listdir(input_dir)])
    fig, axes = plt.subplots(1, 1, figsize=plt.figaspect(0.25), sharex=True)
    return fig, [axes], work_load_list

def prepare_best_draw(input_dir):
    work_load_list = sorted([work_load for work_load in os.listdir(input_dir) if 'MIX' in work_load])
    # work_load_list = ['Helios_MIX1', 'Helios_MIX2', 'Philly_MIX1', 'Philly_MIX2']
    fig, axes = plt.subplots(1, 1, figsize=plt.figaspect(0.5), sharex=True)
    return fig, [axes], work_load_list




def plot_slo_dist(input_dir, logger_info_list, save_path):

    def slo_dist(x, y, z):
        return (x + y) * 1.0 / z

    fig, axes, work_load_list = prepare_draw(input_dir)

    for wid, work_load in enumerate(work_load_list):
        cmap = get_cmap(len(logger_info_list))
        x_list, y_list = list(), list()
        for logger_id, logger_info in enumerate(logger_info_list):
            filename = os.path.join(input_dir, str(work_load), logger_info)
            df = pd.read_csv(filename)
            df = df.drop(df[df.best_effort == 1].index)
            slo_dist_list = [slo_dist(x, y, z) for x, y, z in zip(df['pending_time'].to_list(), df['total_executed_time'].to_list(), df['expect_time'].to_list())]
            x, y = get_cdf(slo_dist_list)
            axes[wid].plot(x, y, label=get_value_from_filename(filename), color=filename_to_color(filename), linewidth=3)
            x_list.append(x)
            y_list.append(y)

        axes[wid].set_ylabel(r'$CDF$', fontsize=fontsize*1.5)    
        axes[wid].set_xlabel(r'$SojournTime/DDLTime$', fontsize=fontsize*1.5)   
        # axes[wid].set_title(r'$workload = {}$'.format(work_load), fontsize=fontsize*1.5)   
        for tick in axes[wid].xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize*1.5)   

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,fancybox=True, shadow=True, fontsize=fontsize*1.5)

    plt.ylim(0, 100)
    plt.xlim(0, 5)
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def box_plot_slo_dist(input_dir, logger_info_list, save_path):
    def slo_dist(x, y, z):
        return (x + y) * 1.0 / z

    fig, axes, work_load_list = prepare_draw(input_dir)

    for wid, work_load in enumerate(work_load_list):
        cmap = get_cmap(len(logger_info_list))
        box_list = list()
        label_list = list()
        for logger_id, logger_info in enumerate(logger_info_list):
            if 'yarn' in logger_info: continue
            filename = os.path.join(input_dir, str(work_load), logger_info)
            df = pd.read_csv(filename)
            df = df.drop(df[df.best_effort == 1].index)
            slo_dist_list = [slo_dist(x, y, z) for x, y, z in zip(df['pending_time'].to_list(), df['total_executed_time'].to_list(), df['expect_time'].to_list())]
            label=get_value_from_filename(filename)
            box_list.append(slo_dist_list)
            label_list.append(label)
            # print(work_load,  label, max(slo_dist_list))
        
        axes[wid].boxplot(box_list, labels=label_list)
        axes[wid].set_ylabel(r'$SLO$')    
        # axes[wid].set_xlabel(r'$(RunningTime+PendingTime)/DDLTime$')
        axes[wid].set_title(r'$workload = {}$'.format(work_load))
        axes[wid].set_ylim(0, 5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,fancybox=True, shadow=True)

    
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')

    

# ---------------------------------------------- print function -------------------------------------------------------------

fontsize = 20
labelsize = 20

def print_job_submit_time(trace_file, save_path):
    cmap = get_cmap(20)
    df = pd.read_csv(trace_file)
    submit_list = df['submit_time']
    gpu_list = df['num_gpu']
    time_hour = [0 for _ in range(24)]
    gpu_hour = [0 for _ in range(24)]
    for idx, x in enumerate(submit_list):
        hour = (x // 60) % 24
        time_hour[hour] += 1. / 14
        gpu_hour[hour] += gpu_list[idx] * 1. / 14

    width = 0.35
    max_hour = 10
    plt.bar([i for i in range(max_hour)], time_hour[:max_hour], width=width, alpha=0.9, color=cmap(15))
    plt.bar([i+width for i in range(max_hour)], gpu_hour[:max_hour],  width=width, alpha=0.4, color=cmap(2))
    labels = ['%02d h'% i for i in range(0, max_hour, 2)]
    plt.xticks([i for i in range(0, max_hour, 2)], labels, rotation='vertical')

    plt.ylabel('Job Number/Request GPU')
    # plt.title('Submit Job Number ')
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def print_job_gpu(trace_file_list, save_path):
    cmap = get_cmap(20)
    for trace_file in trace_file_list:
        if 'MIX' in trace_file: continue 
        df = pd.read_csv(trace_file)
        gpu_list = df['num_gpu']
        gpu_stats = {}
        key_list = ['1', '2', '3-4', '5-8', '>8'] # , '9-16', '>16']
        for key in key_list:
            gpu_stats[key] = 0
        tot = 0
        for gpu in gpu_list:
            tot += 1
            if gpu == 1:
                gpu_stats['1'] += 1
            elif gpu == 2:
                gpu_stats['2'] += 1
            elif gpu <= 4:
                gpu_stats['3-4'] += 1
            elif gpu <= 8:
                gpu_stats['5-8'] += 1
            else:
                gpu_stats['>8'] += 1
            # elif gpu <= 16:
            #     gpu_stats['9-16'] += 1
            # elif gpu > 16:
            #     gpu_stats['>16'] += 1
        
        v_list = [gpu_stats[key] * 1.0 / tot for key in key_list]
        width = 0.4
        delta = width/2 if 'Helios' in trace_file else -1 * width / 2 
        rect = plt.bar([i + delta for i in range(len(key_list))], v_list, width=width, color=trace2color(trace_file), edgecolor='black', label=trace2name(trace_file))
        autolabel(rect, plt)
    plt.xticks([i for i in range(len(key_list))], key_list)
    plt.ylabel('Job Perceptange(%)')
    # plt.title('Submit Job Number ')
    # plt.show()
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')



def tostr(queue):
    if queue == ACCEPTED_SLO:
        return 'accepted'
    elif queue == UNACCEPTED_SLO:
        return 'unaccepted'
    elif queue == BEST_EFFORT:
        return 'best'
    elif queue == LOWEST:
        return 'lowest'
    else:
        raise NotImplementedError

def extract_accepted(queue):
    if tostr(queue) == 'accepted':
        return True
    return False

def extract_other(queue):
    if tostr(queue) not in ['best', 'accepted']:
        return True
    return False



def print_accepted_miss_deadline_percent(input_dir, logger_info_list, save_path):
    work_load_list = ['Helios_SLO', 'Helios_MIX1', 'Helios_MIX2', 'Philly_SLO', 'Philly_MIX1', 'Philly_MIX2']
    show_work_load_list = ['H_SLO', 'H_MIX1', 'H_MIX2', 'P_SLO', 'P_MIX1', 'P_MIX2']
    # work_load_list = sorted([work_load for work_load in os.listdir(input_dir)])
    fig = plt.figure(figsize=(8,6))
    axes = [fig.add_subplot(111)]
    ax2 = axes[0].twinx()    
    axes[0].set_zorder(ax2.get_zorder()-1)
    ax2.patch.set_visible(False) 
    ax2.set_yticks([])


    selected_dir = None
    wid = 0
    width = 0.5

    accepted_miss_list = list() 
    unaccepted_miss_list = list() 
    for method_id, logger_info in enumerate(logger_info_list):
        miss_ddl_list =list()
        method_list = list()
        for _, work_load in enumerate(work_load_list):
            filename = os.path.join(input_dir, str(work_load), logger_info)
            df = pd.read_csv(filename)
            df = df.drop(df[df.best_effort == 1].index)
            # accepted
            df = df[df.queue_priority == ACCEPTED_SLO]
            miss_ddl = df['miss_ddl'].mean() * 100
            accepted_miss_list.append(miss_ddl)
            print('acc', len(df['miss_ddl']))
            # unaccepted
            df = pd.read_csv(filename)
            df = df.drop(df[df.best_effort == 1].index)
            miss_ddl = (df[df.queue_priority == UNACCEPTED_SLO]['miss_ddl'].mean() * 0.1 + df[df.queue_priority == LOWEST]['miss_ddl'].mean() * 0.9) * 100
                
            unaccepted_miss_list.append(miss_ddl)
            print(miss_ddl)

        method_list.append(get_value_from_filename(filename))
        
        
        # ind = [i-width/2 for i in range(len(work_load_list))]
        my_color = filename_to_color(filename)
        # rect = axes[wid].bar(ind, accepted_miss_list, width=width, color=my_color, alpha=0.75, edgecolor='black', label='accepted')
        ind = [i for i in range(len(work_load_list))]
        rect = axes[wid].bar(ind, unaccepted_miss_list, width=width, color=my_color, alpha=0.75, edgecolor='black', label='unaccepted')
        autolabel_percent_int(rect, axes[wid])
        axes[wid].set_xticks([i for i in range(len(work_load_list))])
        axes[wid].set_xticklabels(show_work_load_list, fontsize=fontsize)
        axes[wid].set_ylabel('wDMR (%)', fontsize=fontsize)
        # axes[wid].set_xlabel(r'$workload$', fontsize=fontsize)
        # axes[wid].set_title(r'$workload = {}$'.format(work_load))
        for tick in axes[wid].xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
            # tick.label.set_rotation('vertical')
        axes[wid].tick_params(axis='both', which='major', labelsize=labelsize)
        axes[wid].set_ylim(0,100)
    
    axes[wid].set_xlabel("Workload", fontsize=fontsize)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].grid(axis='y',  linestyle='-.')
    fig.set_size_inches(w=6/0.75, h=6)
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 


def print_miss_deadline_percent(input_dir, logger_info_list, save_path):
    fig, axes, work_load_list = prepare_all_draw(input_dir)
    show_work_load_list = [ 'H_MIX1', 'H_MIX2', 'H_SLO'] # ,  'P_MIX1', 'P_MIX2', 'P_SLO',]
    wid = 0
    width = 0.5

    skip_len = len(logger_info_list) * 0.5 + 1
    mid_pos = len(logger_info_list) * width * 0.5

    for method_id, logger_info in enumerate(logger_info_list):
        miss_ddl_list =list()
        method_list = list()
        for _, work_load in enumerate(work_load_list):
            filename = os.path.join(input_dir, str(work_load), logger_info)
            
            df = pd.read_csv(filename)
            # work_load_length = len(df)
            df = df.drop(df[df.best_effort == 1].index)
            miss_ddl = 100 * (df['miss_ddl']).sum() * 1.0 / len(df)
            if miss_ddl >= 40:
                miss_ddl = (miss_ddl - 40) / 2 + 40
            miss_ddl_list.append(miss_ddl)
            if 'Philly_MIX' in work_load:
                print(logger_info, len(df))

        method_list.append(get_value_from_filename(filename))
        
        method_div = 0 if len(method_list) % 2 == 0 else 0.5
        ind = [i*skip_len+width*method_id + method_div*width for i in range(len(work_load_list))]
        my_color = filename_to_color(filename)
        rect = axes[wid].bar(ind, miss_ddl_list, width=width, color=my_color, alpha=0.75, edgecolor='black', label=method_list[0])
        print(method_list[0], miss_ddl_list)
        autolabel_percent_int(rect, axes[wid])
        axes[wid].set_xticks([mid_pos + i * skip_len for i in range(len(work_load_list))])
        axes[wid].set_xticklabels(show_work_load_list, rotation=0, fontsize=fontsize)
        axes[wid].set_ylabel('wDMR(%)', fontsize=fontsize)
        # axes[wid].set_xlabel(r'$workload$', fontsize=fontsize)
        # axes[wid].set_title(r'$workload = {}$'.format(work_load))
        for tick in axes[wid].xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 

        axes[wid].set_ylim(0, 65)
        axes[wid].set_yticks([0, 5, 10, 15, 20, 40, 55, 65])
        axes[wid].set_yticklabels(['0', '5', '10', '15', '20', '40', '70', '90'], fontsize=fontsize-2)

    axes[wid].set_xlabel('Workload', fontsize=fontsize)
    handles, labels = axes[0].get_legend_handles_labels()
    loc ='upper center'
    fig.legend(handles, labels,  bbox_to_anchor=(1.0, 1.05), ncol=3, fancybox=True, shadow=False, fontsize=fontsize)
    plt.grid(axis='y',  linestyle='-.')
    fig.set_size_inches(w=6/0.75, h=6)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')



def print_best_average_jct(input_dir, logger_info_list, save_path):
    work_load_list = ['Helios_MIX1', 'Helios_MIX2'] # , 'Philly_MIX1', 'Philly_MIX2']
    show_work_load_list = ['H_MIX1', 'H_MIX2'] # , 'P_MIX1', 'P_MIX2']
    fig = plt.figure(figsize=(8,6))
    axes = [fig.add_subplot(111)]
    ax2 = axes[0].twinx()    
    axes[0].set_zorder(ax2.get_zorder()-1)
    ax2.patch.set_visible(False) 
    ax2.set_yticks([])

    wid = 0
    width = 0.4

    skip_len = len(logger_info_list) * 0.5 + 1
    mid_pos = len(logger_info_list) * width * 0.5
    norm_jct_list = list()
    for method_id, logger_info in enumerate(logger_info_list):
        jct_list =list()
        method_list = list()
        
        for _, work_load in enumerate(work_load_list):
            filename = os.path.join(input_dir, str(work_load), logger_info)
            df = pd.read_csv(filename)
            df = df.drop(df[df.best_effort == 0].index)
            aJCT = (df['end_time'] - df['submit_time']).mean()
            jct_list.append(1.0  * aJCT)
            if 'Philly_MIX' in work_load:
                print(logger_info, aJCT)
        method_list.append(get_value_from_filename(filename))
        if len(norm_jct_list) == 0:
            norm_jct_list = jct_list
        show_jct_list = [jct / norm_jct for jct, norm_jct in zip(jct_list, norm_jct_list)]
        jct_list = list()
        for jct in show_jct_list:
            if jct <= 10:
                jct_list.append(jct)
            else:
                jct_list.append((jct - 10) / 2 + 10)

        method_div = 0.5 if len(logger_info_list) % 2 == 0 else 0
        ind = [i*skip_len+width*method_id + method_div * width for i in range(len(work_load_list))]
        my_color = filename_to_color(filename)
        rect = axes[wid].bar(ind, jct_list, width=width, color=my_color, alpha=0.75, edgecolor='black', label=method_list[0])
        print(method_list[0], jct_list)
        autolabel(rect, axes[wid])
        axes[wid].set_xticks([mid_pos + i * skip_len for i in range(len(work_load_list))])
        axes[wid].set_xticklabels(show_work_load_list, fontsize=fontsize)
        axes[wid].set_ylabel('Normalized Average Latency', fontsize=fontsize)
        axes[wid].set_yticks([0, 5, 10, 15, 20])
        axes[wid].set_yticks([0,2,  4, 5, 10, 15, 20])
        axes[wid].set_yticklabels(['0','2', '4','5', '10', '20', '30'], fontsize=fontsize)
        for tick in axes[wid].xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
            # tick.label.set_rotation('vertical')
        axes[wid].set_ylim(0,20)

    axes[wid].set_xlabel('Workload', fontsize=fontsize)
    plt.grid(axis='y',  linestyle='-.')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.0, 1.05), ncol=3, fancybox=True, shadow=False, fontsize=fontsize) # , bbox_to_anchor=(0.5, 0.9))
    # fig.legend(handles, labels,  bbox_to_anchor=(0.9, 1.15), ncol=3, fancybox=True, shadow=False, fontsize=fontsize)
    axes[0].grid(axis='y',  linestyle='-.')
    fig.set_size_inches(w=6/0.75, h=6)
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 


def print_weighted_miss_deadline_percent(input_dir, logger_info_list, save_path):
    fig, axes, work_load_list = prepare_weighted_draw(input_dir)
    wid = 0
    width = 0.2

    skip_len = len(logger_info_list) * 0.5 + 1
    mid_pos = len(logger_info_list) * width * 0.5

    for method_id, logger_info in enumerate(logger_info_list):
        miss_ddl_list =list()
        method_list = list()
        for _, work_load in enumerate(work_load_list):
            filename = os.path.join(input_dir, str(work_load), logger_info)
            
            df = pd.read_csv(filename)
            # work_load_length = len(df)
            df = df.drop(df[df.best_effort == 1].index)
            miss_ddl = 100 * (df['miss_ddl']).sum() * 1.0 / len(df)
            miss_ddl_list.append(miss_ddl)
            if 'Philly_MIX' in work_load:
                print(logger_info, len(df))

        method_list.append(get_value_from_filename(filename))
        method_div = 0 if len(method_list) % 2 == 0 else 0.5
        
        ind = [i*skip_len+width*method_id + method_div*width for i in range(len(work_load_list))]
        my_color = filename_to_color(filename)
        rect = axes[wid].bar(ind, miss_ddl_list, width=width, color=my_color, alpha=0.75, edgecolor='black', label=method_list[0])
        autolabel_percent_int(rect, axes[wid])
        axes[wid].set_xticks([mid_pos + i * skip_len for i in range(len(work_load_list))])
        axes[wid].set_xticklabels(work_load_list, rotation=30)
        axes[wid].set_ylabel('wDMR (%)', fontsize=fontsize)
        # axes[wid].set_xlabel(r'$workload$', fontsize=fontsize)
        # axes[wid].set_title(r'$workload = {}$'.format(work_load))
        for tick in axes[wid].xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
            # tick.label.set_rotation('vertical')
        # axes[wid].tick_params(axis='both', which='major', labelsize=labelsize)
        axes[wid].tick_params(axis='y', which='major', labelsize=10)
        axes[wid].set_ylim(0,100)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fancybox=True, shadow=False) # , bbox_to_anchor=(0.5, 0.9))
    plt.grid(axis='y',  linestyle='-.')
    # plt.show()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')



def print_latency(input_dir, logger_info_list, save_path):
    work_load_list = ['Philly_MIX1', 'Philly_MIX2', 'Philly_SLO', 'Helios_MIX1', 'Helios_MIX2','Helios_SLO']

    wid = 0
    width = 0.5

    skip_len = len(logger_info_list) * 0.5 + 1
    mid_pos = len(logger_info_list) * width * 0.5
    box_list = list()
    label_list = list() 

    for method_id, logger_info in enumerate(logger_info_list):
        for _, work_load in enumerate(work_load_list):
            filename = os.path.join(input_dir, str(work_load), logger_info)
            latency_list = np.load(filename)
            latency_list = [item for item in latency_list if item < 20]
            box_list.append(latency_list)
            label_list.append(work_load)
    my_fontsize=16
    color='blue'
    # plt.legend(handles, labels, loc='upper center', ncol=4, fancybox=True, shadow=False) # , bbox_to_anchor=(0.5, 0.9))
    plt.boxplot(box_list,  widths=0.3, showfliers=False, medianprops={'color':'red'}, boxprops=dict({'color':color}), capprops = {'color': color}, whiskerprops={'color': color})
    plt.xticks([i+1 for i in  range(len(work_load_list))], work_load_list, fontsize=my_fontsize, rotation=30)
    plt.yticks([i*2 for i in  range(7)], [i*2 for i in  range(7)], fontsize=10, rotation=0)
    plt.ylabel('Solver  Latency(s)', fontsize=my_fontsize)
    plt.grid(axis='y',  linestyle='-.')
    plt.ylim(0, 12)
    # plt.show()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def plot_submission(trace_file_list, save_path):
    factor = 60
    my_fontsize=16
    div_hour = 60
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    for trace_file in trace_file_list:
        # if 'sh40' not in trace_file:
        #     continue 

        df = pd.read_csv(trace_file)
        submit_list = df['submit_time'].to_list()
        gpu_list = df['num_gpu'].to_list() 

        max_submit = max(submit_list) // (factor * div_hour)
        x_list = [i for i in range(max_submit + 1)]
        y_list = [0 for i in range(max_submit + 1)]
        for i in range(len(submit_list)):
            idx = submit_list[i] // (factor * div_hour)
            y_list[idx] += gpu_list[i]
        ax.plot(x_list, y_list, linewidth=3, label=trace2name(trace_file), color=trace2color(trace_file))
    # plt.xticks([i * 24 * 60 // div_hour for i in range(16)], [str(i) for i in range(16)])
    plt.xticks([i * 24 * 60 // div_hour for i in [0, 5, 10, 14]], [str(i) for i in [0, 5, 10, 14]])

    delta=5
    plt.tick_params(axis='y', which='major', labelsize=fontsize+delta)
    plt.tick_params(axis='x', which='major', labelsize=fontsize+delta)
    plt.yscale('log')
    
    plt.xlabel('Day', fontsize=fontsize+delta)
    plt.ylabel('Request GPU Number Per Hour',fontsize =fontsize+delta)
    plt.legend(fontsize=fontsize+delta)
    plt.xlim(0, 14 * 24 * 60 // div_hour)
    
    plt.ylim(1, 10**4)
    plt.grid(axis='y',  linestyle='-.')
    fig.set_size_inches(w=6/0.75, h=6.1)
    plt.savefig(save_path, dpi=300)
    plt.close('all')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_dir', type=str, help='input file directory')
    parser.add_argument('--output_file_dir', type=str, help='output file directory')
    logger_list = [
        'advanced-time-aware-with-lease.csv',
        'sigma.csv',
        # 'time-aware-with-lease.csv',
        # 'tetri-sched.csv',
        'genie.csv',
        'time-aware.csv',
        'dlas.csv',
        # 'srtf.csv',
        'themis.csv',
        # 'yarn-cs.csv',
        # 'fifo.csv', 
    ]

    args = parser.parse_args()
    print(args.output_file_dir)
    if not os.path.exists(args.output_file_dir):
        os.makedirs(args.output_file_dir)
    # print_latency(args.input_file_dir, ['time-aware-with-leaselatency_info.npy'], os.path.join(args.output_file_dir, 'latency.png'))

    print_miss_deadline_percent(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'ddl_percent.png'))
    print_best_average_jct(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'bestJCT.png'))
    # print_accepted_miss_deadline_percent(args.input_file_dir, ['time-aware-with-lease.csv'], os.path.join(args.output_file_dir, 'unaccepted_ddl.png'))
    
    trace_file_list = [
        'data/data/dense_trace_cluster_sh40.csv', 
        'data/philly/philly_trace_seconds.csv'
        # 'data/exp/trace_job_Philly_MIX2.csv', 
        # 'data/exp/trace_job_Helios_MIX2.csv', 
    ]

    # plot_job_duration_distribution(trace_file_list, os.path.join(args.output_file_dir, 'job_duration_cdf.png'))
    # plot_submission(trace_file_list, os.path.join(args.output_file_dir, 'profile_submission.png'))
    # print_job_gpu(trace_file_list, os.path.join(args.output_file_dir, 'job_gpu.png'))
    # plot_job_ddl_duration_correlation(trace_file_list, os.path.join(args.output_file_dir, 'job_ddl_duration_correlation.png'))
    

if __name__ == "__main__":
    main()
