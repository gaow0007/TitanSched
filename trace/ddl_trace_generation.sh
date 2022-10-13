# simple version
# python calibrate.py --min_time=300 --max_time=36000 --num_jobs=320 --add_ddl=False --add_user=False --add_job_name=False
node=79
prefix="srun --nodes=1 --gres=gpu:0 --cpus-per-task=4 --ntasks=1 -w SG-IDC1-10-51-2-$node"

for config in MIX1 MIX2 SLO
do 
    $prefix python -u calibrate.py --min_time=300 --max_time=36000 --num_jobs=320 --add_ddl=True --add_user=False --add_job_name=False --ddl_yaml=ddl_config/$config.yaml --add_fm=False --save_root=DDL/$config
done