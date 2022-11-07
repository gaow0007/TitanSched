# trace generation for all workloads
# python calibrate.py --min_time=300 --max_time=36000 --num_jobs=320 --add_ddl=False --add_user=False --add_job_name=False
# node=77
# prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=4 --ntasks=1 -w SG-IDC1-10-51-2-$node"
# $prefix python -u calibrate.py --full_trace=True --save_root=all_trace/ --min_time=-1 --max_time=-1 --num_jobs=320 --add_ddl=False --add_user=False --add_job_name=False --add_fm=True --add_norm=False


node=76
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=4 --ntasks=1 -w SG-IDC1-10-51-2-$node"

config=FM


for model_pair in "roberta-base 120"  "roberta-large 60" # "vit 320" "vit-large 160"
do
    a=( $model_pair )
    model=${a[0]}
    base=${a[1]}

    for data_number in 6 9 
    do
        add_metric=True
        $prefix python -u trace/data_calibrate.py --model=$model --min_time=300 --max_time=36000 \
        --add_metric=$add_metric --add_ddl=False --add_user=False --add_job_name=False --add_fm=False \
        --add_norm=False --repeat_number=1 --save_root=trace/Data/FM-$model-${data_number}/  \
        --data_number=$data_number --density=1 --base=$base
    done 
done

