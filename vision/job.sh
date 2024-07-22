#!/bin/bash
#SBATCH --job-name=wtmk_img_attack
#SBATCH --account=None
#SBATCH --output=None
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00

# Boilerplate environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

wtmk_name='stable-signature'
echo ${wtmk_name}
python batch_attack.py --scheme=${wtmk_name} --batch_size=40 --sample_size=200 --mask_ratio=0.02 --attack_steps=200 --save_folder=hpsv2_results/turbo