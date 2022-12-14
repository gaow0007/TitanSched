node=75
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=4 --ntasks=1 -w SG-IDC1-10-51-2-$node"

# trace generation 
# bash trace/all_trace_generation.sh

for root in  'trace/main'
do 
    # match="FM-"
    match="FM-1-"
    # for trace in `{ls $root/FM-* `
    for trace in `ls trace/main`
    # for trace in FM-160-roberta-base
    # for trace in FM-480-vit
    # for trace in debug
    # for trace in FM-480-vit
    do  
        if [[ "$trace" == *"$match"*  ]]; then 
            echo $trace 
            num_node_p_switch=8
            num_gpu_p_node=4
            
            add_ckpt=30
            
            for failure_ratio in 0 5 10 # 20
            do 
                for schedule in  titan  tiresias optimus pollux # titan pollux tiresias optimus
                do 
                    extra_cmd=""
                    scheduling_time_interval=300
                    ident="0_failure_${schedule}_${trace}_${failure_ratio}"
                    save_log_dir=result/physical/$schedule/$trace/${failure_ratio}
                    mkdir -p $save_log_dir
                    multi_task_adaptivity=True

                    if [[ $schedule == "titan" ]] ;
                    then 
                        temporal_transferability=True
                        transferability=True
                        extra_cmd=" --multi_task_adaptivity=$multi_task_adaptivity --temporal_transferability=$temporal_transferability --transferability=$transferability"
                        scheduling_time_interval=120
                    fi 

                    if [[ $schedule == "themis" ]] ;
                    then 
                        # extra_cmd="--multi_task_adaptivity"
                        extra_cmd=" --lease_term_interval=300"
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
                                --physical=True --failure_ratio=$failure_ratio \
                                --job_type=$job_type --add_ckpt=$add_ckpt ${extra_cmd} &
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
