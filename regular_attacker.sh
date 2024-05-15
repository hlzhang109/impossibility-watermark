#!/bin/bash

# Check if five arguments were provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <cuda_device> <prompt_num> <output_num> <attack_id> <num_successful_steps>"
    exit 1
fi

# Assign the command line arguments to variables
cuda_device=$1
prompt_num=$2
output_num=$3
attack_id=$4
num_successful_steps=$5

results_dir="/local1/borito1907/impossibility-watermark/results/stationary_distribution/robustness_analysis/entropy_$prompt_num/output_$output_num/corpuses"
save_name="attack_$attack_id"
cmdline_output_path="/local1/borito1907/impossibility-watermark/results/stationary_distribution/robustness_analysis/entropy_$prompt_num/output_$output_num/cmdline_output/$save_name.txt"

# Start the command string
cmd="CUDA_VISIBLE_DEVICES=\"$cuda_device\" python -m attack "

# Append parts of the command for readability
cmd+="attack_args.watermarked_text_path=\"./inputs/round_${prompt_num}_outputs.csv\" "
cmd+="attack_args.watermarked_text_num=$output_num "
cmd+="attack_args.results_dir=$results_dir "
cmd+="attack_args.save_name=\"${save_name}.csv\" "
cmd+="attack_args.prompt_num=$prompt_num "
cmd+="attack_args.num_successful_steps=$num_successful_steps "

# Redirect output to a file specified by cmdline_output_path
cmd+="&> \"$cmdline_output_path\""

# Print the command to verify it's correct (or execute it directly)
echo "$cmd"
