# simple version
# python calibrate.py --min_time=300 --max_time=36000 --num_jobs=320 --add_ddl=False --add_user=False --add_job_name=False
node=77
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=4 --ntasks=1 -w SG-IDC1-10-51-2-$node"
$prefix python -u calibrate.py --full_trace=True --save_root=all_trace/ --min_time=-1 --max_time=-1 --num_jobs=320 --add_ddl=False --add_user=False --add_job_name=False --add_fm=True --add_norm=False
