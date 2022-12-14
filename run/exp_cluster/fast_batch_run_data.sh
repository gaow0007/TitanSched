node=76
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=4 --ntasks=1 -w SG-IDC1-10-51-2-$node"

# trace generation 
# bash trace/all_trace_generation.sh

for root in  'trace/Data/'
do 
    # match="FM-"
    match="FM-"
    # for trace in `{ls $root/FM-* `
    for trace in `ls trace/Data/`
    # for trace in FM-roberta-base-9
    # for trace in FM-480-vit
    # for trace in debug
    # for trace in FM-320-vit
    do  
        if [[ "$trace" == *"$match"*  ]]; then 
            echo $trace 
            num_node_p_switch=8
            num_gpu_p_node=4
            
            add_ckpt=30
            for multi_task_adaptivity in True False
            do 
                for schedule in titan
                do 
                    extra_cmd=""
                    scheduling_time_interval=300
                    ident="data_${schedule}_${trace}_${multi_task_adaptivity}"
                    save_log_dir=result/data/$trace/$schedule-${multi_task_adaptivity}
                    if [[ $schedule != "titan" &&  $multi_task_adaptivity == "True" ]]; 
                    then 
                        continue 
                    fi 

                    
                    if [[ $schedule == "titan" ]] ;
                    then 
                        temporal_transferability=True
                        transferability=True
                        extra_cmd=" --multi_task_adaptivity=$multi_task_adaptivity --temporal_transferability=$temporal_transferability --transferability=$transferability"
                        # ident="${schedule}_${trace}_${multi_task_adaptivity}_${transferability}"
                        # save_log_dir=result/$schedule/$trace-${multi_task_adaptivity}-${transferability}
                        scheduling_time_interval=120
                    fi 

                    if [[ $schedule == "themis" ]] ;
                    then 
                        scheduling_time_interval=60
                        # extra_cmd="--multi_task_adaptivity"
                        extra_cmd=" --lease_term_interval=600  "
                    fi 

                    if [[ $schedule == "gavel" ]] ;
                    then 
                        scheduling_time_interval=60
                        # extra_cmd="--multi_task_adaptivity"
                        extra_cmd=" --lease_term_interval=600 "
                    fi 

                    job_type="foundation_model"
                    if [[ $schedule == "pollux" ]] ; 
                    then 
                        job_type="batch_elastic"
                        scheduling_time_interval=300
                    fi 

                    $prefix python -u main.py --schedule=$schedule --trace=$root/$trace/workload-0.csv \
                                --save_log_dir=${save_log_dir} --ident=$ident \
                                --placement=consolidate --num_node_p_switch=$num_node_p_switch \
                                --num_gpu_p_node=$num_gpu_p_node --scheduling_time_interval=$scheduling_time_interval \
                                --job_type=$job_type --add_ckpt=$add_ckpt ${extra_cmd} &
                done
            done 
        fi
        
    done
done 

wait 
srun --nodes=1 --gres=gpu:0 --cpus-per-task=8 --ntasks=1 -w SG-IDC1-10-51-2-76 python plot/draw_data.py

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
