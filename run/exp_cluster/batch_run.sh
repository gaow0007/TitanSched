node=75
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=4 --ntasks=1 -w SG-IDC1-10-51-2-$node"

# trace generation 
# bash trace/all_trace_generation.sh

for root in  'trace/main/'
do 
    # match="FM-"
    match="FM-"
    # for trace in `{ls $root/FM-* `
    # for trace in `ls trace/`
    # for trace in FM-720-vit-large
    # for trace in FM-480-vit
    # for trace in debug
    for trace in FM-1-vit-large # FM-480-roberta-base
    do  
        if [[ "$trace" == *"$match"*  ]]; then 
            echo $trace 
            num_node_p_switch=8
            num_gpu_p_node=4
            scheduling_time_interval=120
            add_ckpt=30
            for multi_task_adaptivity in True False  # True False 
            do 
                for schedule in titan #  themis  tiresias optimus srtf # titan # themis  tiresias optimus srtf  # tiresias optimus srtf  # srtf # themis # titan tiresias optimus srtf 
                do 
                    extra_cmd=""
                    ident="${schedule}_${trace}"
                    save_log_dir=result/$schedule/$trace
                    if [[ $schedule != "titan" &&  $multi_task_adaptivity == "True" ]]; 
                    then 
                        continue 
                    fi 

                    
                    if [[ $schedule == "titan" ]] ;
                    then 
                        # 0.391540 (hour) for normal scheduling 
                        # 1.164363 for normal scheduling
                        temporal_transferability=True
                        extra_cmd=" --multi_task_adaptivity=$multi_task_adaptivity --transferability=$temporal_transferability" # 0.302186
                        ident="${schedule}_${trace}_${multi_task_adaptivity}"
                        save_log_dir=result/$schedule/$trace-${multi_task_adaptivity}
                        # extra_cmd=""
                        scheduling_time_interval=120
                    fi 

                    if [[ $schedule == "themis" ]] ;
                    then 
                        # extra_cmd="--multi_task_adaptivity"
                        extra_cmd=" --lease_term_interval=600"
                    fi 

                    job_type="foundation_model"
                    if [[ $schedule == "pollux" ]] ; 
                    then 
                        job_type="batch_elastic"
                    fi 

                    echo $job_type
                    $prefix python -u main.py --schedule=$schedule --trace=$root/$trace/workload-0.csv \
                                --save_log_dir=${save_log_dir} --ident=$ident \
                                --placement=consolidate --num_node_p_switch=$num_node_p_switch \
                                --num_gpu_p_node=$num_gpu_p_node --scheduling_time_interval=$scheduling_time_interval \
                                --job_type=$job_type --add_ckpt=$add_ckpt ${extra_cmd}  &
                done
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
