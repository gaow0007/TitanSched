node=75
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=1 --ntasks=1 -w SG-IDC1-10-51-2-$node"

 # 'trace/DDL/MIX2/' 'trace/DDL/SLO/' # 'trace/min-300-max-36000-num-320'

for root in  'trace/norm-720/'
do 
    for trace in 'Helios' 
    do  
        echo $trace
        num_node_p_switch=24
        num_gpu_p_node=4

        # for schedule in srtf 
        # do 
        # mkdir -p result/$schedule/
        #     $prefix python main.py --schedule=$schedule --trace=$root/"$trace".csv --save_log_dir=result/$schedule/"$trace"_"$num_queue" --ident="$schedule"_"$trace" \
        #             --placement=consolidate --num_node_p_switch=$num_node_p_switch --scheduling_time_interval=30 --job_type='preempt'
        # done 

        # $prefix python main.py --schedule=$schedule --trace=$root/"$trace".csv --save_log_dir=result/$schedule/"$trace"_"$num_queue" --ident="$schedule"_"$trace"_"$num_queue" \
        #         --placement=consolidate --num_node_p_switch=$num_node_p_switch --scheduling_time_interval=30 --job_type='preempt' \
        #         --num_queue=$num_queue --queue_limit 3600 1209600 --search_algorithm=history --hpo_search_time_interval=900

        for schedule in tiresias
        do 
            mkdir -p result/$schedule/
            # for num_queue in 2 3 4 5 6 7
            for num_queue in 2
            do 
                $prefix python main.py --schedule=$schedule --trace=$root/"$trace".csv --save_log_dir=result/$schedule/"$trace"_"$num_queue" --ident="$schedule"_"$trace"_"$num_queue"_history \
                        --placement=consolidate --num_node_p_switch=$num_node_p_switch --scheduling_time_interval=30 --job_type='preempt' \
                        --num_queue=$num_queue --queue_limit 3600 1209600 --search_algorithm=history --hpo_search_time_interval=3600

                # $prefix python main.py --schedule=$schedule --trace=$root/"$trace".csv --save_log_dir=result/$schedule/"$trace"_"$num_queue" --ident="$schedule"_"$trace"_"$num_queue" \
                #         --placement=consolidate --num_node_p_switch=$num_node_p_switch --scheduling_time_interval=30 --job_type='preempt' \
                #         --num_queue=$num_queue --queue_limit 3600 1209600
            done 
        done 
    done
done 
