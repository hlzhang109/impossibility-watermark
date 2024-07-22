#!/bin/bash
#SBATCH --job-name=wtmk_text_attack
#SBATCH --account=None
#SBATCH --output=None
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00

wtmk_name='KGW' 
attack_name='Random-Walk' 
model_name_or_path="meta-llama/Meta-Llama-3-70B-Instruct" # "meta-llama/Llama-2-7b"

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model_name_or_path=${model_name_or_path} \
               --wtmk_name=${wtmk_name} \
               --attack_name=${attack_name} \
               --max_new_tokens=300 \
               --num_samples=13 \
               --save_folder="results"
