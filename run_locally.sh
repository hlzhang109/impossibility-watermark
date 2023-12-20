TIMESTAMP=$(date +"%Y%m%d%H%M%S")
python3 attack.py --input "short_input.csv" --output "./results/output_$TIMESTAMP.csv" --step_T 10 --num_trials 2 --check_quality True