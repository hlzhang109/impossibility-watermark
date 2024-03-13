#!/bin/bash

# Check if six arguments were provided
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <cuda_device> <prompt_num> <output_num> <attack_id> <new_attack_id> <num_successful_steps>"
    exit 1
fi

# Assign the command line arguments to variables
cuda_device=$1
prompt_num=$2
output_num=$3
attack_id=$4
new_attack_id=$5
num_successful_steps=$6

results_dir="/home/borito1907/impossibility-watermark/results/stationary_distribution/robustness_analysis/entropy_$prompt_num/output_$output_num/corpuses"
prev_csv_file="attack_$attack_id"
save_name="attack_$new_attack_id"
cmdline_output_path="/home/borito1907/impossibility-watermark/results/stationary_distribution/robustness_analysis/entropy_$prompt_num/output_$output_num/cmdline_output/$save_name.txt"


# Start the command string
cmd="CUDA_VISIBLE_DEVICES=$cuda_device python -m attack "

# Append parts of the command for readability
cmd+="attack_args.prev_csv_file=\"${prev_csv_file}.csv\" "
cmd+="attack_args.results_dir=$results_dir "
cmd+="attack_args.save_name=\"${save_name}.csv\" "
cmd+="attack_args.is_continuation=True "
cmd+="attack_args.prompt_num=$prompt_num "
cmd+="attack_args.num_successful_steps=$num_successful_steps "

# Redirect output to a file specified by cmdline_output_path
cmd+="&> \"$cmdline_output_path\""

# Print the command to verify it's correct (or execute it directly)
echo "$cmd"
