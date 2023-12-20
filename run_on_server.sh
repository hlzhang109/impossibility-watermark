TIMESTAMP=$(date +"%Y%m%d%H%M%S")
python attack.py --input "test_input.csv" --output "./results/output_$TIMESTAMP.csv" --step_T 500 --num_trials 4 --check_quality True