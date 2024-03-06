#!/bin/bash

# Check if three arguments were provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <cuda_device> <index> <save_name>"
    exit 1
fi

# Assign the command line arguments to variables
cuda_device=$1
i=$2
save_name=$3

CUDA_VISIBLE_DEVICES=$cuda_device python -m attack attack_args.json_index="$i" attack_args.save_name="$save_name" attack_args.is_completion=True attack_args.json_path='./text_completions_50_c4.json' &> completion_"$i".txt
