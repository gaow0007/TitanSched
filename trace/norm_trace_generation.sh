# simple version
# python calibrate.py --min_time=300 --max_time=36000 --num_jobs=320 --add_ddl=False --add_user=False --add_job_name=False
node=77
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=4 --ntasks=1 -w SG-IDC1-10-51-2-$node"

# for config in FM
# do 
#     $prefix python -u calibrate.py --min_time=300 --max_time=36000 --num_jobs=320 --add_ddl=False --add_user=False --add_job_name=False --add_fm=True --fm_yaml=fm_config/FM.yaml --save_root=FM/ 
# done

for density in 720
do 
    $prefix python -u calibrate.py --min_time=300 --max_time=36000 --num_jobs=$density --add_ddl=False --add_user=False --add_job_name=False --add_fm=False --add_norm=True --norm_yaml=norm_config/norm.yaml --save_root=norm-$density/ 
done