def generate_prefix(): 
    return """
export OMP_NUM_THREADS=4
prefix=""
if type srun >/dev/null 2>&1;
then 
    username=$(whoami)
    if [[ $username == "wgao" ]]; then 
        prefix="srun --nodes=1 --gres=gpu:4 --cpus-per-task=12 --ntasks=1 -w SG-IDC1-10-51-2-76"
    fi 
    if [[ $username == "sunpeng" ]]; then 
        prefix="srun -p caif_debug -n 1 --preempt --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=12"
    fi 
fi
"""
