node=75
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=12 --ntasks=1 -w SG-IDC1-10-51-2-$node"

# trace generation 
# bash trace/all_trace_generation.sh

for root in  'trace/'
do 
    match="FM-"
    # for trace in `{ls $root/FM-* `
    for trace in `ls trace/`
    # for trace in FM-160-roberta-base
    do  
        if [[ "$trace" == *"$match"*  ]]; then 
            echo $trace 
            num_node_p_switch=16
            num_gpu_p_node=4

            for schedule in  titan # optimus tiresias srtf # titan # titan # chockwave optimus tiresias titan srtf 
            do 
                extra_cmd=""
                if [[ $schedle == "titan" ]] ;
                then 
                    # extra_cmd="--multi_task_adaptivity"
                    extra_cmd=""
                fi 
                job_type="foundation_model"
                $prefix python -u main.py --schedule=$schedule --trace=$root/$trace/workload-0.csv \
                            --save_log_dir=result/$schedule/$trace --ident=$schedule_$trace \
                            --placement=consolidate --num_node_p_switch=$num_node_p_switch \
                            --num_gpu_p_node=$num_gpu_p_node --scheduling_time_interval=300 --job_type=$job_type ${extra_cmd}
            done
        fi
        
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
