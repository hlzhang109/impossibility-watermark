TIMESTAMP=$(date +"%Y%m%d%H%M%S")
python3.10 attack.py --input "short_input.csv" --output "./results/output_$TIMESTAMP" --step_T 3 --mask_filling_model_name "google/t5-v1_1-small" --num_trials 3
