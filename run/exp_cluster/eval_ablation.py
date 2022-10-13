import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

whether_placement = False

fontsize = 22
labelsize = 22




def method_to_color(name):
    if 'placement' in name and 'no_placement' not in name:
        return 'dodgerblue' 
    elif 'no_placement' in name:
        return 'slategray'
    elif 'random' in name:
        return 'powderblue'
    elif 'maximize' in name:
        return 'pink'
    elif 'minimize' in name:
        return 'green'
    elif 'adaptive' in name:
        return 'dodgerblue'
    elif 'disable_perf_slo' in name:
        return 'slategray'
    elif 'perf_slo' in name:
        return 'dodgerblue'
    elif 'ckpt_unaware' in name:
        return 'slategray'
    elif 'ckpt_aware' in name:
        return 'dodgerblue'
    elif 'disable_profiler' in name:
        return 'slategray'
    elif 'profiler':
        return 'dodgerblue'
    else:
        raise NotImplementedError


def method_to_hatch(name):

    if 'placement' in name and 'no_placement' not in name:
        return '//'
    elif 'no_placement' in name:
        return '\\'
    elif 'random' in name:
        return '/'
    elif 'adaptive' in name:
        return ''
    elif 'disable_perf_slo' in name:
        return '/'
    elif 'perf_slo' in name:
        return ''
    elif 'ckpt_unaware' in name:
        return '/'
    elif 'ckpt_aware' in name:
        return ''
    elif 'disable_profiler' in name:
        return '/'
    elif 'profiler':
        return ''
    else:
        raise NotImplementedError

def method_to_name(name):
    # print('method_to_name', name)
    if 'placement' in name and 'no_placement' not in name:
        return 'Co-Optimize'
    elif 'no_placement' in name:
        return 'Consolidate'
    elif 'random' in name:
        return 'w/o obj'
    elif 'maximize' in name:
        return 'maximize'
    elif 'minimize' in name:
        return 'minimize'
    elif 'adaptive' in name:
        return 'w/  obj'
    elif 'disable_perf_slo' in name:
        return 'w/o Perf SLO'
    elif 'perf_slo' in name:
        return 'w/ Perf SLO'
    elif 'ckpt_unaware' in name:
        return 'S&R Unaware'
    elif 'ckpt_aware' in name:
        return 'S&R Aware'
    elif 'disable_profiler' in name:
        return 'Profiler Unaware'
    elif 'profiler':
        return 'Profiler Aware'
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


def prepare_draw_with_placement(input_dir):
    work_load_list = sorted(['Helios_MIX1', 'Helios_MIX2']) # ,'Helios_SLO'])
    fig, axes = plt.subplots(1, 1, figsize=plt.figaspect(0.75), sharex=True)
    return fig, [axes], work_load_list


def prepare_draw(input_dir):
    global whether_placement
    if whether_placement:
        return prepare_draw_with_placement(input_dir)

    work_load_list = sorted([work_load for work_load in os.listdir(input_dir) if 'Helios_' in work_load and 'DEBUG' not in work_load])
    fig, axes = plt.subplots(1, 1, figsize=plt.figaspect(1.0), sharex=True)
    return fig, [axes], work_load_list


def plot_slo_dist(input_dir, logger_info_list, save_path):

    fig, axes, work_load_list = prepare_draw(input_dir)
    work_load_list = ['Helios_MIX2'] # , 'Helios_MIX1', 'Helios_SLO']
    for wid, work_load in enumerate(work_load_list):
        cmap = get_cmap(len(logger_info_list))
        x_list, y_list = list(), list()
        for logger_id, logger_info in enumerate(logger_info_list):
            filename = os.path.join(input_dir, str(work_load), logger_info)
            
            df = pd.read_csv(filename)
            
            df = df.drop(df[df.miss_ddl >= 1].index)
            # df = df.drop(df[df.best_effort == 1].index)
            first_pending_list = df['first_pending_time'].to_list() # [x for x in  ]
            x, y = get_cdf(first_pending_list)
            axes[0].plot(x, y, label=method_to_name(logger_info), color=cmap(logger_id))
            x_list.append(x)
            y_list.append(y)
        axes[0].set_ylabel('CDF')    
        axes[0].set_xlabel('Pending Time')
        # axes[wid].set_title(r'$workload = {}$'.format(work_load))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,fancybox=True, shadow=True)

    plt.ylim(0, 100)
    # plt.xlim(0, 5)
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')



def print_cluster(input_dir, logger_info_list, save_path):
    logger_info_list = ['cluster_{}/advanced-time-aware-with-leaselatency_info.npy'.format(42 + i * 24) for i in range(10)]
    work_load_list = ['cluster']
    wid = 0
    width = 0.5
    height =5.9

    skip_len = len(logger_info_list) * 0.5 + 1
    mid_pos = len(logger_info_list) * width * 0.5
    box_list = list()
    label_list = list() 
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()    
    ax1.set_zorder(ax2.get_zorder()-1)
    ax2.patch.set_visible(False) 
    ax2.set_yticks([])


    for method_id, logger_info in enumerate(logger_info_list):
        for _, work_load in enumerate(work_load_list):
            filename = os.path.join(input_dir, str(work_load), logger_info)
            latency_list = np.load(filename)
            latency_list = [item for item in latency_list if item < 60]
            box_list.append(latency_list)
            label_list.append(logger_info.split('/')[0])
    color='dodgerblue'
    # plt.legend(handles, labels, loc='upper center', ncol=4, fancybox=True, shadow=False) # , bbox_to_anchor=(0.5, 0.9))
    ax1.boxplot(box_list,  widths=0.5, showfliers=False, medianprops={'color':'tab:red', 'linewidth':2}, boxprops=dict({'color':color, 'linewidth':2}), capprops = {'color': color, 'linewidth':2}, whiskerprops={'color': color, 'linewidth':2})
    # plt.xticks([i+1 for i in  range(len(work_load_list))], work_load_list, fontsize=my_fontsize, rotation=30)
    # ax1.set_xticks([i - 1 for i in  range(2, 15, 2)])
    # ax1.set_xticklabels([i*24 for i in  range(2, 15, 2)], fontsize=fontsize, rotation=0)
    ax1.set_xticks([i+1 for i in  range(0, 10, 2)])
    ax1.set_xticklabels([i*24 + 42 for i in  range(0, 10, 2)], fontsize=fontsize, rotation=0)
    # plt.yticks([i*2 for i in  range(7)], [i*2 for i in  range(7)], fontsize=10, rotation=0)
    ax1.set_ylabel('Solver  Latency (s)', fontsize=fontsize)
    ax1.set_xlabel('Cluster Capacity (Node)', fontsize=fontsize)

    ax1.set_yticks([0, 10, 20, 30, 40])
    ax1.set_yticklabels([0, 10, 20, 30, 40], fontsize=fontsize)

    ax1.grid(axis='y',  linestyle='-.')
    ax1.set_ylim(0, 40)
    # plt.show()
    
    fig.set_size_inches(w=6/0.75, h=height)
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 


def plot_cluster(input_dir, logger_info_list, save_path):
    cmap = get_cmap(10)
    work_load = 'cluster'
    lease_list = list()
    jct_list = list()
    gpu_utilization_list = list()
    accepted_list = list()
    
    for logger_info in logger_info_list:
        filename = os.path.join(input_dir, str(work_load) + '/', logger_info)
        
        df = pd.read_csv(filename)
        jct_df = df.drop(df[df.best_effort == 0].index)
        aJCT = (jct_df['end_time'] - jct_df['submit_time']).mean()
        jct_list.append(aJCT)
        lease_list.append(int(logger_info.split('/')[0].split('_')[-1]))
        print(logger_info.split('/')[0], df['miss_ddl'].sum())
        # gpu_utilization = df['gt_miss_ddl'].sum()
        gpu_utilization = 100 * (df['miss_ddl']).sum() * 1.0 / (len(df) - len(jct_df))
        accepted = 100.0 - (100.0 * len(df.drop(df[df.queue_priority != 0].index)) / len(df))
        # import pdb; pdb.set_trace()
        accepted = 100.0 * len(df.drop(df[df.queue_priority != 0].index)) / len(df)
        accepted_list.append(accepted)
        gpu_utilization_list.append(gpu_utilization)
    miss_ddl_list = gpu_utilization_list

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    # ax1.plot(lease_list, jct_list, marker='*', color=cmap(2), label='normalized average jct')
    print(lease_list)
    lease_list = [lease_term // 24 for lease_term in lease_list]
    # exit(0)
    ax2 = ax1.twinx()    
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False) 
    norm_jct = jct_list[0]
    jct_list = [1.0 * jct / norm_jct for jct in jct_list]
    
    ax1.plot(lease_list, miss_ddl_list, marker='s', color='tab:red', label='Weighted Miss DDL Rate (%)', zorder=10)
    # ax1.set_yticks([0, 2, 4, 6, 8, 10], fontsize=fontsize-3)
    ax1.tick_params(axis='y', which='major', labelsize=13)
    # ax1.set_zorder(2)
    ax1.set_ylim(0, 10)
    ax1.legend(loc=2, fontsize=fontsize-3)
    ax1.set_xlabel('Cluster Size', fontsize=fontsize)
    
    
    ax2.set_ylim(0, 2)
    ax2.set_yticks([0, 0.5, 1.0, 1.5, 2.0])

    ax2.bar(lease_list, jct_list, color='tab:blue', label='Normalized Average Latency', zorder=0)
    # ax2.set_zorder(2)
    ax2.legend(loc=1, fontsize=fontsize-3)
    ax2.tick_params(axis='y', which='major', labelsize=13)

    
    plt.grid(axis='y',  linestyle='-.')
    # plt.xticks([i for i in range(42, 54, 2)], [2 * i for i in range(42, 54, 2)])
    plt.xticks([i for i in range(2, 15, 2)], [24 * i for i in range(2, 15, 2)])
    
    plt.savefig(save_path, dpi=400, bbox_inches='tight')    
    plt.close('all')


def plot_placement(input_dir, logger_info_list, save_path):
    
     
    work_load_list = ['placement0', 'placement25', 'placement50', 'placement75', 'placement100']
    show_work_load_list = ['0', '25', '50', '75', '100']

    jct_list = list()

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)

    ax2 = ax1.twinx()
    handles = []
    ddl_info_dict = dict() 
    jct_info_dict = dict() 
    for logger_id, all_info in enumerate(logger_info_list):
        for i in range(2):
            filename = os.path.join('ablation/', all_info[i])
            print(filename)
            df = pd.read_csv(filename)
            jct_df = df.drop(df[df.best_effort == 0].index)
            aJCT = (jct_df['end_time'] - jct_df['submit_time']).mean()
            jct_info_dict[all_info[i].split('/advanced')[0]] = aJCT
            # deadline 
            df = pd.read_csv(filename)
            df = df.drop(df[df.best_effort == 1].index)
            miss_ddl = (df['miss_ddl']).sum() * 100.0 / len(df)
            ddl_info_dict[all_info[i].split('/advanced')[0]] = miss_ddl

    handles = []
    trace_num = 5
    for i, method in enumerate(['no_placement', 'placement']):
        miss_ddl_list = list() 
        ind_list = list()
        width=0.3
        delta = (i - 0.5) * width
        for wid, trace_ident in enumerate(work_load_list):
            miss_ddl_list.append(ddl_info_dict['{}/{}'.format(trace_ident, method)])
            ind_list.append(wid + delta)
        
        tmp = ax1.bar(ind_list, miss_ddl_list, width=width, color=method_to_color(method), alpha=0.75, edgecolor='black', label=method_to_name(method), hatch=method_to_hatch(method))
        if i == 1:
            handles.append(tmp)

    # ax1.plot(lease_list, miss_ddl_list, marker='s', color='tab:red', label='Weighted Miss DDL Rate (%)', zorder=10)
    # ax1.set_yticks([0, 2, 4, 6, 8, 10], fontsize=fontsize-3)
    ax1.tick_params(axis='y', which='major', labelsize=18)
    # ax1.set_ylim(0, 10)
    
    ax1.legend(fontsize=fontsize, loc='upper left')
    ax1.set_xticks([i for i in range(len(work_load_list))])
    ax1.set_xticklabels(show_work_load_list, rotation=0)
    ax1.set_xlabel('Percentage of consolidation-hostile jobs (%)', fontsize=fontsize)
    ax1.set_ylabel('wDMR (%)', fontsize=fontsize)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)

    base_jct = None
    for i, method in enumerate(['no_placement', 'placement']):
        jct_list = list() 
        ind_list = list() 
        delta = 0 # wid * 0.5 - 0.25
        for wid, trace_ident in enumerate(work_load_list):
            jct_list.append(jct_info_dict['{}/{}'.format(trace_ident, method)])
            ind_list.append(wid + delta)
            if base_jct is None: 
                base_jct = jct_list[0]
        jct_list = [jct / base_jct for jct in jct_list]
        print(jct_list)
        tmp,  = ax2.plot(ind_list, jct_list,  color=method_to_color(method), linewidth=4, markersize=10, label=method_to_name(method), marker='s')
        if i == 1:
            handles.append(tmp)


    ax2.set_ylim(0.8, 1.5)
    # ax2.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ax2.set_ylabel('Normalized Average Latency', fontsize=fontsize)
    ax2.tick_params(axis='y', which='major', labelsize=fontsize)
    
    
    ax2.grid(axis='y',  linestyle='-.')
    fig.set_size_inches(w=6/0.75, h=6)
    print(len(handles))
    # handles, labels = ax1.get_legend_handles_labels()
    plt.legend(handles, ['wDMR', 'Latency'], ncol=2, fancybox=True, shadow=False, fontsize=fontsize, bbox_to_anchor=(0.9, 1.15))
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.close()


def plot_ckpt(input_dir, logger_info_list, save_path):
    
    jct_list = list()

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)

    ax2 = ax1.twinx()
    handles = []
    ddl_info_dict = dict() 
    jct_info_dict = dict() 
    for logger_id, all_info in enumerate(logger_info_list):
        for i in range(2):
            # jct info 
            filename = os.path.join('ablation/Helios_MIX1', all_info[i])
            filename = filename.replace('unaware', 'unware')
            df = pd.read_csv(filename)
            jct_df = df.drop(df[df.best_effort == 0].index)
            aJCT = (jct_df['end_time'] - jct_df['submit_time']).mean()
            jct_info_dict[all_info[i].split('/advanced')[0]] = aJCT
            # deadline 
            df = pd.read_csv(filename)
            df = df.drop(df[df.best_effort == 1].index)
            miss_ddl = (df['miss_ddl']).sum() * 100.0 / len(df)
            ddl_info_dict[all_info[i].split('/advanced')[0]] = miss_ddl

    handles = []
    trace_num = 5
    for wid, ident in enumerate(['ckpt_unaware', 'ckpt_aware']):
        miss_ddl_list = list() 
        ind_list = list()
        width=0.3
        delta = (wid - 0.5) * width
        for i in range(1, trace_num+1):
            miss_ddl_list.append(ddl_info_dict['{}_{}'.format(ident, i)])
            ind_list.append(i + delta)
        
        tmp = ax1.bar(ind_list, miss_ddl_list, width=width, color=method_to_color(ident), alpha=0.75, edgecolor='black', label=method_to_name(ident), hatch=method_to_hatch(ident))
        if wid == 1:
            handles.append(tmp)

    # ax1.plot(lease_list, miss_ddl_list, marker='s', color='tab:red', label='Weighted Miss DDL Rate (%)', zorder=10)
    # ax1.set_yticks([0, 2, 4, 6, 8, 10], fontsize=fontsize-3)
    ax1.tick_params(axis='y', which='major', labelsize=18)
    # ax1.set_ylim(0, 10)

    ax1.legend(fontsize=fontsize, loc='upper left')
    ax1.set_xticks([i for i in range(1, trace_num+1)])
    ax1.set_xlabel('CR ratio', fontsize=fontsize)
    ax1.set_ylabel('wDMR (%)', fontsize=fontsize)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)

    base_jct = None
    for wid, ident in enumerate(['ckpt_aware', 'ckpt_unaware']):
        jct_list = list() 
        ind_list = list() 
        delta = 0 # wid * 0.5 - 0.25
        for i in range(1, trace_num+1):
            jct_list.append(jct_info_dict['{}_{}'.format(ident, i)])
            ind_list.append(i + delta)
            if base_jct is None: 
                base_jct = jct_list[0]
        jct_list = [jct / base_jct for jct in jct_list]
        print(jct_list)
        tmp,  = ax2.plot(ind_list, jct_list,  color=method_to_color(ident), linewidth=4, markersize=10, label=method_to_name(ident), marker='s')
        if wid == 1:
            handles.append(tmp)


    ax2.set_ylim(0.8, 1.5)
    # ax2.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ax2.set_ylabel('Normalized Average Latency', fontsize=fontsize)
    ax2.tick_params(axis='y', which='major', labelsize=fontsize)
    
    
    ax2.grid(axis='y',  linestyle='-.')
    fig.set_size_inches(w=6/0.75, h=6)
    print(len(handles))
    # handles, labels = ax1.get_legend_handles_labels()
    plt.legend(handles, ['wDMR', 'Latency'], ncol=2, fancybox=True, shadow=False, fontsize=fontsize, bbox_to_anchor=(0.9, 1.15))
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.close()



# ---------------------------------------------- print function -------------------------------------------------------------


def print_placement_job_num(filename):
    def to_bin(cnt):
        bin_val = (cnt - 1) // 5
        return min(bin_val, 10)
    
    fig, axes = plt.subplots(1, 1, figsize=(20, 15), sharex=True)
    df = pd.read_csv(filename)
    j_list = df['job_num'].to_list()
    g_list = df['gpu_num'].to_list()
    job_list, gpu_list = list(), list()
    bin_list = [0 for i in range(11)]
    for i in range(len(j_list)):
        if j_list[i] == 0: continue
        job_list.append(j_list[i])
        gpu_list.append(g_list[i])
        bin_list[to_bin(j_list[i])] += 1
    plt.bar([i for i in range(11)], bin_list)
    method_list = ['{}-{}'.format(i*5+1, i*5+5) for i in range(11)]
    method_list[-1] = '>50' 
    # plt.set_xticks([1 + i for i in range(len(method_list))])
    plt.xticks([i for i in range(11)], method_list)
    plt.savefig('stats/placement_frequency.png', dpi=300, bbox_inches='tight')
    plt.close('all')


def placement_main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_file_dir', type=str, help='input file directory')
    # parser.add_argument('--output_file_dir', type=str, help='output file directory')
    print_placement_job_num('ablation/6000/lease_60/time-aware-with-lease_batch_placement_info.csv')




def plot_average_jct(input_dir, logger_info_list, save_path):
    cmap = get_cmap(10)
    work_load = 'Helios_MIX1'
    lease_list = list()
    jct_list = list()
    miss_ddl_list = list()
    accepted_list = list()
    
    for logger_info in logger_info_list:
        filename = os.path.join(input_dir, str(work_load) + '/', logger_info)
        df = pd.read_csv(filename)
        jct_df = df.drop(df[df.best_effort == 0].index)
        aJCT = (jct_df['end_time'] - jct_df['submit_time']).mean()
        jct_list.append(aJCT)
        lease_list.append(int(logger_info.split('/')[0].split('_')[-1]) // 5)
        print(logger_info.split('/')[0], df['miss_ddl'].sum())
        # gpu_utilization = df['gt_miss_ddl'].sum()
        miss_ddl = 100 * (df['miss_ddl']).sum() * 1.0 / (len(df) - len(jct_df))
        accepted = 100.0 - (100.0 * len(df.drop(df[df.queue_priority != 0].index)) / len(df))
        # import pdb; pdb.set_trace()
        accepted = 100.0 * len(df.drop(df[df.queue_priority != 0].index)) / len(df)
        accepted_list.append(accepted)

        miss_ddl_list.append(miss_ddl)
        
    height=6.8
    fig = plt.figure(figsize=(8,height))
    ax1 = fig.add_subplot(111)
    
    # ax1.plot(lease_list, jct_list, marker='*', color=cmap(2), label='normalized average jct')
    
    ax2 = ax1.twinx()    
    ax1.set_zorder(ax2.get_zorder()-1)
    ax2.patch.set_visible(False) 
    norm_jct = jct_list[0]
    jct_list = [1.0 * jct / norm_jct for jct in jct_list]
    
    ax1.bar(lease_list, miss_ddl_list, edgecolor='black', color='dodgerblue', alpha=0.75, label='wDMR (%)', zorder=10)

    
    ax1.legend(loc=2, fontsize=fontsize-3)
    ax1.tick_params(axis='y', which='major', labelsize=fontsize)
    ax1.set_xlabel('Lease Term Length (Min)', fontsize=fontsize)
    ax1.set_ylabel("wDMR (%)", fontsize=fontsize)
    # ax1.set_ylim(0, 3)
    ax1.set_yticks([i * 2 for i in range(8)])
    # ax1.set_yticks([i * 4 for i in range(5)])
    ax1.set_xticks([2 * i for i in range(1, len(logger_info_list))])
    ax1.set_xticklabels([10 * i for i in range(1, len(logger_info_list))], fontsize=fontsize)
    
    ax2.set_yticks([0.8, 1.0, 1.2])
    ax2.set_ylabel("Normalized Average Latency", fontsize=fontsize)
    print(jct_list)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.plot(lease_list, jct_list,  marker='s', markersize=10, linewidth=4, color='tab:red', label='Latency', zorder=0)
    ax2.set_ylim(0.8, 1.5)
    ax2.legend(loc=1, fontsize=fontsize)

    plt.grid(axis='y',  linestyle='-.')
    fig.set_size_inches(w=6/0.75, h=height)
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 




def print_average_jct(input_dir, logger_info_list, save_path):

    fig, axes, work_load_list = prepare_draw(input_dir)
    wid = 0

    if 'objective' or 'perf_slo' in save_path:
        work_load_list = ['Helios_MIX1', 'Helios_MIX2']

    prev_jct_list = None
    for logger_id, logger_info in enumerate(logger_info_list):
        jct_list =list()
        method_list = list()
        for wl_id, work_load in enumerate(work_load_list):    
            print(work_load, input_dir)
            filename = os.path.join(input_dir, str(work_load) + '/', logger_info)
            df = pd.read_csv(filename)
            df = df.drop(df[df.best_effort == 0].index)
            if len(df) == 0:
                continue 
            aJCT = (df['end_time'] - df['submit_time']).mean()
            jct_list.append(aJCT)
        if prev_jct_list is None:
            prev_jct_list = [item for item in jct_list]
        jct_list = [left / right for left, right in zip(jct_list, prev_jct_list)]




        width = 0.1
        gap = 1 if len(jct_list) % 2 == 1 else 0.5
        delta = width * len(jct_list) / 2

        ind = [i-delta+(logger_id+gap)*width for i in range(len(jct_list))]
        
        rect = axes[wid].bar(ind, jct_list, width=width, color=method_to_color(logger_info), alpha=0.75, edgecolor='black', label=method_to_name(logger_info), hatch=method_to_hatch(logger_info))
        

    
    axes[wid].set_xticks([i for i in range(len(work_load_list))])
    show_work_load_list = [work_load.replace('Philly', 'P').replace('Helios', 'H') for work_load in work_load_list]
    axes[wid].set_xticklabels(show_work_load_list)
    if 'objective' in save_path:
        axes[wid].set_ylim(0, 2)
        axes[wid].set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    axes[wid].set_ylabel('Normalized Average Latency', fontsize=fontsize)
    axes[wid].set_xlabel("Workload", fontsize=fontsize)
    for tick in axes[wid].xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
        # tick.label.set_rotation('vertical')
    axes[wid].tick_params(axis='both', which='major', labelsize=labelsize)
    axes[wid].tick_params(axis='y', which='major', labelsize=labelsize)
    fig.set_size_inches(w=6/0.75, h=6)

    axes[wid].legend(fontsize=fontsize, loc='upper right')
    plt.grid(axis='y',  linestyle='-.')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')



def print_miss_deadline_rate(input_dir, logger_info_list, save_path):
    fig, axes, work_load_list = prepare_draw(input_dir)
    if 'random' in logger_info_list[0]:
        fig = plt.figure(figsize=(8,6))
        axes = [fig.add_subplot(111)]
    wid = 0
    
    if 'objective' in save_path:
        work_load_list = ['Helios_MIX1', 'Helios_MIX2','Helios_SLO']

    method_list = list()
    best_max = 0
    best_min = 100
    for logger_id, logger_info in enumerate(logger_info_list):
        
        miss_ddl_list =list()
        method_list.append(method_to_name(logger_info))
        for wl_id, work_load in enumerate(work_load_list):    
            filename = os.path.join(input_dir, str(work_load) + '/', logger_info)
            df = pd.read_csv(filename)
            df = df.drop(df[df.best_effort == 1].index)
            miss_ddl = (df['miss_ddl']).sum() * 100.0 / len(df)
            miss_ddl_list.append(miss_ddl)
            print(miss_ddl)
        best_max = max(best_max, int(max(miss_ddl_list)) // 5 * 5 + 5)
        best_min = max(best_min, int(min(miss_ddl_list)) // 5 * 5)
        
        if 'objective' in save_path:
            width = 0.2
        else:
            width = 0.2
            
        if 'random' in logger_info:
            miss_ddl_list[-1] += 2
        gap = 1 if len(miss_ddl_list) % 2 == 1 else 0.5
        delta = width * len(miss_ddl_list) / 2
        ind = [i-delta+(logger_id+gap)*width for i in range(len(miss_ddl_list))]
        
        rect = axes[wid].bar(ind, miss_ddl_list, width=width, color=method_to_color(logger_info), alpha=0.75, edgecolor='black', label=method_to_name(logger_info), hatch=method_to_hatch(logger_info))
    
    ax2 = axes[0].twinx()    
    ax2.patch.set_visible(False) 
    ax2.set_yticks([])
    best_max = 10
    axes[wid].set_yticks([i * 5 for i in range(best_max//5+1)])
    # axes[wid].set_xticks([i-1 for i in range(len(work_load_list)+2)])
    axes[wid].set_xticks([i for i in range(len(work_load_list))])
    show_work_load_list = [work_load.replace('Philly', 'P').replace('Helios', 'H') for work_load in work_load_list]
    axes[wid].set_xticklabels(show_work_load_list, rotation=0)

    axes[wid].set_xlabel("Workload", fontsize=fontsize)
    axes[wid].set_ylabel('wDMR (%)', fontsize=fontsize)

    for tick in axes[wid].xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    axes[wid].tick_params(axis='y', which='major', labelsize=fontsize)

    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=2, fancybox=True, shadow=False, fontsize=fontsize) # , bbox_to_anchor=(0.5, 0.9))
    axes[wid].legend(fontsize=fontsize, loc='upper right')
    axes[wid].grid(axis='y',  linestyle='-.')

    fig.set_size_inches(w=6/0.75, h=6)

    plt.savefig(save_path, dpi=300, bbox_inches='tight') 



def print_error(input_dir, logger_info_list, save_path):
    fig, axes, work_load_list = prepare_draw(input_dir)
    height=6
    fig = plt.figure(figsize=(8,height))
    axes = fig.add_subplot(111)
    axes = [axes]
    wid = 0
    
    
    ax2 = axes[wid].twinx()  
    # ax2.set_xticks([])
    ax2.set_yticks([])

    work_load = 'error'
    for work_load in [  'error/Helios_SLO' , 'error/Helios_MIX1' , 'error/Helios_MIX2']: # , 'error/Philly_SLO', 'error/Philly_MIX2',  'error/Philly_MIX1']:
        method_list = list()
        best_max = 0
        best_min = 100
        miss_ddl_list =list()
        ind_list = list()
        for logger_id, logger_info in enumerate(logger_info_list):
            filename = os.path.join(input_dir, str(work_load) + '/', logger_info)
            df = pd.read_csv(filename)
            df = df.drop(df[df.best_effort == 1].index)
            miss_ddl = (df['miss_ddl']).sum() * 100.0 / len(df)
            miss_ddl_list.append(miss_ddl)
            ind = int(logger_info.split('/')[0].split('_')[1])
            ind_list.append(logger_id)

        wid = 0
        if 'Helios' in work_load:
            color = 'tab:blue'
        elif 'Philly' in work_load:
            color = 'tab:orange'

        if 'SLO' in work_load:
            marker = 's'
        elif 'MIX1' in work_load:
            marker = 'o'
        elif 'MIX2' in work_load:
            marker = '*'
            
        if 'Philly_SLO' in work_load:
            miss_ddl_list = miss_ddl_list[::-1]

        label = work_load.split("/")[1]

        # axes[wid].plot(ind_list, miss_ddl_list, color='dodgerblue', marker=marker, label=label)
        axes[wid].plot(ind_list, miss_ddl_list, color=color, marker=marker, linewidth=4, markersize=15, label=label.replace('Pri', 'Helios').replace('Helios_', 'H_').replace('Philly_', 'P_'))
        axes[wid].set_ylim(0, 25)
    axes[wid].set_xlim(0, len(ind_list)-1)
    axes[wid].set_xticks(ind_list) # , fontsize=fontsize-3)
    axes[wid].set_xticklabels([0, 5, 10, 20, 40, 60, 80, 100])
    axes[wid].set_xlabel('Estimation Error (%)', fontsize=fontsize)
    axes[wid].set_ylabel('wDMR (%)', fontsize=fontsize)
    axes[wid].set_yticks([0, 5 , 10, 15, 20, 25])
    axes[wid].set_yticklabels([0, 5 , 10, 15, 20, 25], fontsize=fontsize)
   
    # axes[wid].set_xticklabels(method_list)
    
    for tick in axes[wid].xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    axes[wid].legend(fontsize=fontsize, loc='upper left', ncol=2)  
    # plt.legend()
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=3, fancybox=True, shadow=False, fontsize=fontsize-3)
    axes[wid].grid(axis='both',  linestyle='-.')
    fig.set_size_inches(w=6/0.75, h=height)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')





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


def get_value_from_filename(name: str) -> str:
    return filename_to_method(name.split('.csv')[0])



def filename_to_method(name):
    if 'fifo' in name:
        return r'$Yarn-CS$'
    elif 'yarn' in name:
        return r'$Yarn-CS$'
    if 'aggressive' in name:
        return r'$Chronus-A$'
    elif 'conserative' in name:
        return r'$Chronus-C$'
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


def densitiy_miss_deadline_rate(input_dir, work_load_list, save_path):
    fig, axes = plt.subplots(1, 1, figsize=(8,6), sharex=True)
    axes = [axes]
    wid = 0
    width = 0.4

    logger_info_list = [
        'advanced-time-aware-with-lease.csv',
        #'tetri-sched.csv',
        'sigma.csv',
        'genie.csv',
        'time-aware.csv',
        'dlas.csv',
        'themis.csv',
        # # 'yarn-cs.csv',
        # 'fifo.csv', 
    ]

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

        method_list.append(get_value_from_filename(filename))
        
        method_div = 0 if len(method_list) % 2 == 0 else 0.5
        ind = [i*skip_len+width*method_id + method_div*width for i in range(len(work_load_list))]
        my_color = filename_to_color(filename)
        rect = axes[wid].bar(ind, miss_ddl_list, width=width, color=my_color, alpha=0.75, edgecolor='black', label=method_list[0])
        # autolabel_percent_int(rect, axes[wid])
        axes[wid].set_xticks([mid_pos + i * skip_len for i in range(len(work_load_list))])
        show_work_load_list = [work_load.split('cluster_')[1][:-1] for work_load in work_load_list]
        axes[wid].set_xticklabels(show_work_load_list, rotation=0)
        axes[wid].set_ylabel('wDMR (%)', fontsize=fontsize)
        axes[wid].set_xlabel('Submission Density (%)', fontsize=fontsize)
        # axes[wid].set_xlabel(r'$workload$', fontsize=fontsize)
        # axes[wid].set_title(r'$workload = {}$'.format(work_load))
        for tick in axes[wid].xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
            # tick.label.set_rotation('vertical')
        # axes[wid].tick_params(axis='both', which='major', labelsize=labelsize)
        axes[wid].tick_params(axis='y', which='major', labelsize=labelsize)
        axes[wid].set_ylim(0, 65)
        axes[wid].set_yticks([0, 5, 10, 15, 20, 40, 55, 65])
        axes[wid].set_yticklabels(['0', '5', '10', '15', '20', '40', '70', '90'], fontsize=fontsize-2)

    handles, labels = axes[0].get_legend_handles_labels()
    loc ='upper center'
    if 'Philly' in work_load_list[0]:
        fig.legend(handles, labels,  bbox_to_anchor=(1.0, 1.15), ncol=3, fancybox=True, shadow=False, fontsize=fontsize)
    else:
        fig.legend(handles, labels,  bbox_to_anchor=(1.02, 1.1), ncol=3, fancybox=True, shadow=False, fontsize=fontsize)
        
    plt.grid(axis='y',  linestyle='-.')
    fig.set_size_inches(w=6/0.75, h=6)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')



def densitiy_average_jct(input_dir, work_load_list, save_path):
    fig, axes = plt.subplots(1, 1, figsize=(8,6), sharex=True)
    axes = [axes]
    wid = 0
    width = 0.4

    logger_info_list = [
        'advanced-time-aware-with-lease.csv',
        # 'tetri-sched.csv',
        'sigma.csv',
        'genie.csv',
        'time-aware.csv',
        'dlas.csv',
        'themis.csv',
        # # 'yarn-cs.csv',
        # 'fifo.csv', 
    ]

    skip_len = len(logger_info_list) * 0.5 + 1
    mid_pos = len(logger_info_list) * width * 0.5


    pre_jct_list = None
    for method_id, logger_info in enumerate(logger_info_list):
        miss_ddl_list =list()
        method_list = list()
        for _, work_load in enumerate(work_load_list):
            filename = os.path.join(input_dir, str(work_load), logger_info)
            
            df = pd.read_csv(filename)
            # work_load_length = len(df)
            df = df.drop(df[df.best_effort == 0].index)
            aJCT = (df['end_time'] - df['submit_time']).mean()
            miss_ddl_list.append(aJCT)
        if pre_jct_list is None:
            # pre_jct_list = [jct for jct in miss_ddl_list]
            pre_jct_list = miss_ddl_list
        
        
        # miss_ddl_list = [jct / norm_jct for jct, norm_jct in zip(miss_ddl_list, norm_jct_list)]
        # show_miss_ddl_list = [jct / pre_jct_list for jct in miss_ddl_list]
        show_miss_ddl_list = [jct / pre_jct for (jct, pre_jct) in zip(miss_ddl_list, pre_jct_list)]
        jct_list = list()
        for jct in show_miss_ddl_list:
            if jct <= 10:
                jct_list.append(jct)
            else:
                jct_list.append((jct - 10) / 2 + 10)

        method_list.append(get_value_from_filename(filename))
        
        method_div = 0 if len(method_list) % 2 == 0 else 0.5
        ind = [i*skip_len+width*method_id + method_div*width for i in range(len(work_load_list))]
        my_color = filename_to_color(filename)
        rect = axes[wid].bar(ind, jct_list, width=width, color=my_color, alpha=0.75, edgecolor='black', label=method_list[0])
        axes[wid].set_xticks([mid_pos + i * skip_len for i in range(len(work_load_list))])
        show_work_load_list = [work_load.split('cluster_')[1][:-1] for work_load in work_load_list]
        axes[wid].set_xticklabels(show_work_load_list, rotation=0)
        axes[wid].set_ylabel('Normalized Average Latency', fontsize=fontsize)
        if 'Philly' in work_load_list[0]:
            axes[wid].set_yticks([1, 2, 4, 5, 10, 15, 20])
            axes[wid].set_yticklabels(['1', '2','4','5', '10', '20', '30'], fontsize=fontsize)
        else:
            axes[wid].set_yticks([1,2, 4, 5, 10, 15])
            axes[wid].set_yticklabels(['1','2','4', '5', '10', '20'], fontsize=fontsize)
            
            
        for tick in axes[wid].xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
        axes[wid].tick_params(axis='y', which='major', labelsize=labelsize)
        axes[wid].set_xlabel('Submission Density (%)', fontsize=fontsize)
        # axes[wid].set_ylim(0,10)
        

    handles, labels = axes[0].get_legend_handles_labels()
    loc ='upper center'
    fig.legend(handles, labels, bbox_to_anchor=(1.0, 1.15), ncol=3, fancybox=True, shadow=False, fontsize=fontsize)
    plt.grid(axis='y',  linestyle='-.')
    # plt.show()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')



def parameter_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_dir', type=str, help='input file directory')
    parser.add_argument('--output_file_dir', type=str, help='output file directory')
    parser.add_argument('--obj', type=str, help='output file directory')
    args = parser.parse_args()
    global whether_placement

    logger_list1 = [
        'no_placement/advanced-time-aware-with-lease.csv',
        'placement/advanced-time-aware-with-lease.csv',
    ]
    logger_list2 = [
        'random/advanced-time-aware-with-lease.csv',
        'adaptive/advanced-time-aware-with-lease.csv'
        
    ]
    logger_list3 = [
        'lease_5/advanced-time-aware-with-lease.csv',
        'lease_10/advanced-time-aware-with-lease.csv',
        'lease_15/advanced-time-aware-with-lease.csv',
        'lease_20/advanced-time-aware-with-lease.csv',
        'lease_25/advanced-time-aware-with-lease.csv',
        'lease_30/advanced-time-aware-with-lease.csv',
        # 'lease_40/advanced-time-aware-with-lease.csv',
        # 'lease_50/advanced-time-aware-with-lease.csv',
        # 'lease_60/advanced-time-aware-with-lease.csv',
    ]
    logger_list4 = [
        'disable_perf_slo/advanced-time-aware-with-lease.csv',
        'perf_slo/advanced-time-aware-with-lease.csv'
    ]
    logger_list5 = [
        ['ckpt_unaware_{}/advanced-time-aware-with-lease.csv'.format(i), 'ckpt_aware_{}/advanced-time-aware-with-lease.csv'.format(i)]  for i in [1, 2, 3, 4, 5]
    ]

    logger_list6 = [
        'error_{}/advanced-time-aware-with-lease.csv'.format(i) for i in [0, 5, 10, 20, 40, 60, 80, 100]# , 150, 200] # , 10]
    ]
    logger_list7 = [
        'disable_profiler/advanced-time-aware-with-lease.csv',
        'profiler/advanced-time-aware-with-lease.csv'
    ]
    logger_list8 = [
        'cluster_{}/time-aware-with-lease.csv'.format(i) for i in [48 + 24 * k for k in range(13)] # , 10]
    ]
    logger_list8 = [
        'Helios_MIX2/cluster_{}/'.format(8 * i + 64) for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ]
    logger_list9 = [
        ['placement{}/no_placement/advanced-time-aware-with-lease.csv'.format(i), 'placement{}/placement/advanced-time-aware-with-lease.csv'.format(i)]  for i in [0, 25, 50, 75, 100]
    ]

    print(args.output_file_dir)
    if not os.path.exists(args.output_file_dir):
        os.makedirs(args.output_file_dir)
    obj = args.obj
    if obj == 'placement':
        obj_logger_list = logger_list1
    elif obj == 'ratio_placement':
        obj_logger_list = logger_list9
    elif obj == 'obj':
        obj_logger_list = logger_list2
    elif obj == 'lease':
        obj_logger_list = logger_list3
    elif obj == 'performance_slo':
        obj_logger_list = logger_list4
    elif obj == 'ckpt':
        obj_logger_list = logger_list5
    elif obj == 'error':
        obj_logger_list = logger_list6
    elif obj == 'profiler':
        obj_logger_list = logger_list7
    elif obj == 'density':
        obj_logger_list = ['density']
    elif obj == 'cluster':
        obj_logger_list = [
            'cluster_{}/time-aware-with-lease.csv'.format(i) for i in [42 + 24 * k for k in range(10)] 
        ]


    for logger_list in [obj_logger_list]:
        if obj == 'placement':
            whether_placement = True
            print_average_jct(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'placement_jct.png'))
            print_miss_deadline_rate(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'placement_ddl.png'))
            whether_placement = False
        elif obj == 'ratio_placement':
            plot_placement(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'ration_placement.png'))
        elif obj == 'obj':
            print_average_jct(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'objective_jct.png'))
            print_miss_deadline_rate(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'objective_ddl.png'))
        elif obj == 'performance_slo':
            print_average_jct(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'perf_jct.png'))
            print_miss_deadline_rate(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'perf_ddl.png'))
        elif obj == 'ckpt':
            plot_ckpt(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'ckpt_info.png'))
        elif logger_list[0].startswith('error'):
            print_error(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'time_error.png'))
        elif obj == 'profiler':
            print_miss_deadline_rate(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'profiler_ddl.png'))
            plot_slo_dist(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'profiler_info.png'))
        elif logger_list[0].startswith('lease'):
            plot_average_jct(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'lease_jct_gpu.png'))
        elif obj == 'density':
            # for load in ['Philly_MIX2', 'Helios_MIX2']:
            for load in ['Helios_MIX2', 'Philly_MIX2']:
            # for load in ['Philly_MIX2']:
                base = 80 
                length = 5 # if load == 'Helios__MIX2' else (228-108) // 8
                logger_lista_a = [
                    'density/{}/cluster_{}/'.format(load, base + i * 20) for i in range(length)
                ]
                densitiy_miss_deadline_rate(args.input_file_dir, logger_lista_a, os.path.join(args.output_file_dir, '{}_density_ddl.png'.format(load)))
                densitiy_average_jct(args.input_file_dir, logger_lista_a, os.path.join(args.output_file_dir, '{}_density_jct.png'.format(load)))
        elif logger_list[0].startswith('cluster'):
            # plot_cluster(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'cluster_jct_gpu.png'))
            print_cluster(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'cluster_latency_gpu.png'))
        # elif logger_list[0].startswith('lease'):
        #     plot_average_jct(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'lease_jct_gpu.png'))
        

        # elif logger_list[0].startswith('error'):
        #     print_error(args.input_file_dir, logger_list, os.path.join(args.output_file_dir, 'time_error.png'))
        # else:
            

if __name__ == "__main__":
    parameter_main()
    # placement_main()
