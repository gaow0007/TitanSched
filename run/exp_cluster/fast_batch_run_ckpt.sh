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
            for add_ckpt in 0 15 30 45 60 75 90
            do 
                extra_cmd=""

                job_type="foundation_model"
                if [[ $schedule == "titan" ]] ;
                then 
                    temporal_transferability=True
                    transferability=True
                    extra_cmd=" --multi_task_adaptivity=$multi_task_adaptivity --temporal_transferability=$temporal_transferability --transferability=$transferability"
                    ident="ckpt_${add_ckpt}_${schedule}_${trace}"
                    save_log_dir=result/ckpt/$trace/$add_ckpt
                    scheduling_time_interval=120
                fi 
                
                $prefix python -u main.py --schedule=$schedule --trace=$root/$trace/workload-0.csv \
                            --save_log_dir=${save_log_dir} --ident=$ident \
                            --placement=consolidate --num_node_p_switch=$num_node_p_switch \
                            --num_gpu_p_node=$num_gpu_p_node --scheduling_time_interval=$scheduling_time_interval \
                            --job_type=$job_type --add_ckpt=$add_ckpt --estimation_error=0 ${extra_cmd} &
            done 
        fi
        
    done
done 

wait 
