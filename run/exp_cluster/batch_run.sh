node=78
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=12 --ntasks=1 -w SG-IDC1-10-51-2-$node"

 # 'trace/DDL/MIX2/' 'trace/DDL/SLO/' # 'trace/min-300-max-36000-num-320'
# for root in 'trace/DDL/MIX2/' 
# for root in  'trace/DDL/MIX1/'
for root in  'trace/FM/'
do 
    for trace in 'BM' # 'Helios' 'MLaas' 'Philly'
    do  
        echo $trace
        num_node_p_switch=16
        num_gpu_p_node=4

        for schedule in pollux # sigma chronus # edf tetri-sched sigma # tetri-sched #  tetri-sched # edf # tetri-sched # edf sigma # srtf edf tiresias # tetri-sched # tetri-sched # edf
        do 
            job_type="preempt"
            if [[ $schedule == "pollux" ]]; then
                job_type="batch_elastic";
            fi 
            $prefix python -u main.py --schedule=$schedule --trace=$root/"$trace".csv --save_log_dir=result/$schedule/$trace --ident=$schedule_$trace \
            --placement=consolidate --num_node_p_switch=$num_node_p_switch --num_gpu_p_node=$num_gpu_p_node --scheduling_time_interval=300 --job_type=$job_type 
        done

        # for schedule in optimus # sigma chronus # edf tetri-sched sigma # tetri-sched #  tetri-sched # edf # tetri-sched # edf sigma # srtf edf tiresias # tetri-sched # tetri-sched # edf
        # do 
        #     job_type="preempt"
        #     if [[ $schedule == "optimus" ]]; then
        #         job_type="resource_elastic";
        #     fi 
        #     $prefix python main.py --schedule=$schedule --trace=$root/"$trace".csv --save_log_dir=result/$schedule/$trace --ident=$schedule_$trace \
        #     --placement=consolidate --num_node_p_switch=$num_node_p_switch --num_gpu_p_node=$num_gpu_p_node --scheduling_time_interval=30 --job_type=$job_type 
        # done
        
        # for schedule in yarn-cs srtf tiresias  # tiresias srtf # srtf #  # dlas srtf # themis dlas srtf fifo
        # do 
        #     $prefix python main.py --schedule=$schedule --trace=$root/"$trace".csv --save_log_dir=result/$schedule/$trace --ident=$schedule_$trace \
        #     --placement=consolidate --num_node_p_switch=$num_node_p_switch --num_gpu_p_node=$num_gpu_p_node  --scheduling_time_interval=30 --job_type="preempt" 
        # done

        # for scheule in titan # themis dlas srtf fifo
        # do 
        #     job_type="batch_elastic"
        #     # job_type="foundation_model"
        #     $prefix python -u main.py --schedule=$scheule --trace=$root/"$trace".csv --save_log_dir=result/$schedule/$trace --ident=$schedule_$trace \
        #     --placement=consolidate --num_node_p_switch=$num_node_p_switch --num_gpu_p_node=$num_gpu_p_node --scheduling_time_interval=30 --job_type=$job_type --multi_task_adaptivity=False
        # done
    done
done 

 # 'trace/DDL/MIX2/' 'trace/DDL/SLO/' # 'trace/min-300-max-36000-num-320'

# for root in  'trace/FM/'
# do 
#     for trace in 'BM' 
#     do  
#         echo $trace
#         num_node_p_switch=16
#         num_gpu_p_node=4
#         for multi_task_adaptivity in True False 
#         do
#             for scheule in titan # titan srtf # themis dlas srtf fifo
#             do 
#                 $prefix python main.py --schedule=$scheule --trace=$root/"$trace".csv --save_log_dir=result/$schedule/$trace --ident=$schedule_$trace \
#                 --placement=consolidate --num_node_p_switch=$num_node_p_switch --scheduling_time_interval=30 --job_type="foundation_model" --multi_task_adaptivity=$multi_task_adaptivity
#             done
#             cp result/BM/titan.csv result/BM/titan_$multi_task_adaptivity.csv
#         done 
#     done
# done 
