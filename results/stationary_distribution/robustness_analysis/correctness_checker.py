import os
import pandas as pd

def check_attack_csv(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Perform checks
    step_num_check = check_step_num_sequence(df)
    quality_preserved_check = check_quality_preserved_logic(df)
    quality_not_preserved_check = check_quality_not_preserved_logic(df)
    
    # Initialize a flag to keep track of the overall verification status
    checks_passed = True
    
    # Check if any of the checks failed
    if not step_num_check:
        print(f"{file_path}: Step number sequence check failed.")
        checks_passed = False
    if not quality_preserved_check:
        print(f"{file_path}: Quality preserved check failed.")
        checks_passed = False
    if not quality_not_preserved_check:
        print(f"{file_path}: Quality not preserved check failed.")
        checks_passed = False
        
    return checks_passed

def check_step_num_sequence(df):
    sequence_check = True
    
    if df.iloc[0]['step_num'] != -1:
        print(f"The first step number isn't -1.")
        sequence_check = False
    
    for i in range(1, len(df)):
        if df.iloc[i]['step_num'] - df.iloc[i-1]['step_num'] != 1:
            print(f"Step number sequence breaks between rows {i-1} and {i}, with step_num {df.iloc[i-1]['step_num']} followed by {df.iloc[i]['step_num']}")
            sequence_check = False
    return sequence_check

def check_quality_preserved_logic(df):
    quality_check = True
    for i in range(len(df) - 1):
        if df.iloc[i]['quality_preserved']:
            if df.iloc[i + 1]['current_text'] != df.iloc[i]['mutated_text']:
                print(f"Quality preservation check failed at step_num: {df.iloc[i]['step_num']}")
                quality_check = False
    return quality_check

def check_quality_not_preserved_logic(df):
    quality_not_preserved_check = True
    for i in range(len(df) - 1):
        if not df.iloc[i]['quality_preserved']:
            if df.iloc[i + 1]['current_text'] != df.iloc[i]['current_text'] and not df.iloc[i + 1]['backtrack']:
                print(f"Quality not preserved at step_num: {df.iloc[i]['step_num']}")
                quality_not_preserved_check = False
    return quality_not_preserved_check


def process_corpuses(start_path):
    for subdir, dirs, files in os.walk(start_path):
        if os.path.basename(subdir) == 'corpuses':
            for file in files:
                if file.startswith('attack_') and file.endswith('.csv'):
                    filepath = os.path.join(subdir, file)
                    
                    check_attack_csv(filepath)
                    
if __name__ == '__main__':
    start_directory = '/home/borito1907/impossibility-watermark/results/stationary_distribution/robustness_analysis'
    process_corpuses(start_directory)
