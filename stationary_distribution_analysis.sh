#!/bin/bash

# Check if one argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <N>"
    exit 1
fi

N=$1  # The number of times to run the loop
CondaEnv=$CONDA_DEFAULT_ENV # The name of the current conda environment to activate in the tmux sessions
cuda=0  # Initial cuda value - This initialization needs to be before the loop starts

for ((i=1; i<=N; i++)); do
    # Create a new tmux session for each command run, detached (-d) and named ("session_$i")
    tmux new-session -d -s "session_$i"
    
    # Use tmux send-keys to activate the conda environment and then run the command
    # Ensure we correctly reference the variables inside the send-keys command
    tmux send-keys -t "session_$i" "conda activate $CondaEnv" C-m
    tmux send-keys -t "session_$i" "python -m attack attack_args.cuda=\'$cuda\' &> 4_evan_2_${i}.txt" C-m

    # Increment cuda for the next iteration, and ensure it loops through the gpus
    ((cuda++))
    if [ $cuda -gt 3 ]; then
        cuda=0
    fi
done

watch -n 3 nvidia-smi
