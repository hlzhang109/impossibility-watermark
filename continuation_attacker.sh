#!/bin/bash

# Check if four arguments were provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <cuda_device> <save_name> <path_to_csv> <prompt_num>"
    exit 1
fi

# Assign the command line arguments to variables
cuda_device=$1
save_name=$2
prev_csv_file=$3
prompt_num=$4

# Run the Python module with the extracted prompt and other arguments
python -m attack attack_args.prev_csv_file="$prev_csv_file.csv" attack_args.cuda=\'$cuda_device$\' attack_args.save_name="$save_name.csv" attack_args.is_continuation=True attack_args.prompt_num=$prompt_num &> "$save_name".txt
