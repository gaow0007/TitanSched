node=76
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=4 --ntasks=1 -w SG-IDC1-10-51-2-$node"

# trace generation 
# bash trace/all_trace_generation.sh

for root in  'trace/main/'
do 
    match="FM-1-"
    for trace in `ls trace/main/`
    # for trace in FM-1-vit-large
    do  
        if [[ "$trace" == *"$match"*  ]]; then 
            echo $trace 
            num_node_p_switch=8
            num_gpu_p_node=4
            add_ckpt=30
            multi_task_adaptivity=True
            schedule=titan
            for error in 0 5 10 20 40 60 80 100 
            do 
                extra_cmd=""

                job_type="foundation_model"
                if [[ $schedule == "titan" ]] ;
                then 
                    temporal_transferability=True
                    transferability=True
                    extra_cmd=" --multi_task_adaptivity=$multi_task_adaptivity --temporal_transferability=$temporal_transferability --transferability=$transferability"
                    ident="error_${error}_${schedule}_${trace}"
                    save_log_dir=result/error/$trace/$error
                    scheduling_time_interval=120
                fi 
                
                $prefix python -u main.py --schedule=$schedule --trace=$root/$trace/workload-0.csv \
                            --save_log_dir=${save_log_dir} --ident=$ident \
                            --placement=consolidate --num_node_p_switch=$num_node_p_switch \
                            --num_gpu_p_node=$num_gpu_p_node --scheduling_time_interval=$scheduling_time_interval \
                            --job_type=$job_type --add_ckpt=$add_ckpt --estimation_error=$error ${extra_cmd} &
            done 
        fi
        
    done
done 

wait 
# srun --nodes=1 --gres=gpu:0 --cpus-per-task=8 --ntasks=1 -w SG-IDC1-10-51-2-76 python plot/draw_density.py

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
