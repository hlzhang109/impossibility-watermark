#!/bin/bash

# Check if the correct number of arguments were provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <cuda_device> <index> <save_name> <attack_num>"
    exit 1
fi

# Assign command line arguments to variables for clarity
cuda_device=$1
index=$2
save_name=$3
attack_num=$4

# Formulate the save_name according to the provided arguments
final_save_name="${save_name}_${index}_${attack_num}"

# Define the path to the output file
output_path="completion_${index}_${attack_num}.txt"

cmd="CUDA_VISIBLE_DEVICES=\"$cuda_device\" python -m attack "

cmd+="attack_args.cuda='$cuda_device' "
cmd+="attack_args.json_index=\"$index\" "
cmd+="attack_args.save_name=\"${final_save_name}\" "
cmd+="attack_args.is_completion=True "
cmd+="attack_args.json_path='./text_completions_50_c4.json' "

# Redirect output to a file specified by output_path
cmd+="&> \"$output_path\""

echo "$cmd"
