node=76
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=1 --ntasks=1 -w SG-IDC1-10-51-2-$node"

# trace generation 
# bash trace/all_trace_generation.sh
set -e
for root in  'trace/HPO/'
do 
    # match="FM-"
    match="FM-"
    # for trace in `{ls $root/FM-* `
    # for trace in `ls trace/`
    # for trace in FM-720-vit-large
    # for trace in FM-roberta-base
    # for trace in debug
    # for trace in FM-vit # FM-480-roberta-base
    for trace in `ls trace/HPO/`
    do  
        if [[ "$trace" == *"$match"*  ]]; then 
            echo $trace 
            num_node_p_switch=2
            num_gpu_p_node=4
            add_ckpt=30
            for repeat in {1..50} # {1..50} # {1..50} # {31..31}
            do 
                for temporal_transferability in True False # False # False # True 
                do 
                    for schedule in hpo_titan # titan # themis  tiresias optimus srtf  # tiresias optimus srtf  # srtf # themis # titan tiresias optimus srtf 
                    do 
                        extra_cmd=""
                        ident="${schedule}_${trace}"
                        save_log_dir=result/HPO/$schedule/$trace
                        multi_task_adaptivity=False
                        
                        extra_cmd=" --multi_task_adaptivity=$multi_task_adaptivity --temporal_transferability=$temporal_transferability" # 0.302186
                        ident="repeat_${repeat}_${schedule}_${trace}_temporal_transfer_${temporal_transferability}"
                        save_log_dir=result/$schedule/$trace-${temporal_transferability}/$repeat/
                        # extra_cmd=""
                        scheduling_time_interval=120
                        job_type="hpo_foundation_model"
                        # echo $save_log_dir
                        # exit 
                        $prefix python -u main.py --schedule=$schedule --trace=$root/$trace/workload-$repeat.csv \
                                    --save_log_dir=${save_log_dir} --ident=$ident \
                                    --placement=consolidate --num_node_p_switch=$num_node_p_switch \
                                    --num_gpu_p_node=$num_gpu_p_node --scheduling_time_interval=$scheduling_time_interval \
                                    --job_type=$job_type --add_ckpt=$add_ckpt ${extra_cmd} &
                    done
                done 
            done
        fi
        
    done
done 
srun --nodes=1 --gres=gpu:0 --cpus-per-task=8 --ntasks=1 -w SG-IDC1-10-51-2-76 python plot/draw_hpo.py