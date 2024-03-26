import os
import pandas as pd

def update_csv_remove_rows_before_last_step_num_negative_one(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Find the index of the last row where step_num is -1
    index = df[df['step_num'] == -1].index.max()
    
    # Check if such a row exists
    if pd.isna(index):
        print("No row with 'step_num' == -1 found.")
        return df  # Return original DataFrame if no -1 step_num found
    
    # Delete every row before the last occurrence of step_num == -1
    updated_df = df.loc[index:]
    
    # Reset the index of the updated DataFrame
    updated_df.reset_index(drop=True, inplace=True)
    
    updated_df.to_csv(file_path, index=False)
    print(f"CSV file {file_path} updated: rows before first 'step_num' == -1 have been removed.")
    
    return updated_df

def process_corpuses(start_path):
    for subdir, dirs, files in os.walk(start_path):
        if os.path.basename(subdir) == 'corpuses':
            for file in files:
                if file.startswith('attack_') and file.endswith('.csv'):
                    parts = file.split('_')
                    if len(parts) == 3:  # File named like attack_x_y.csv
                        attack_x_y_file_path = os.path.join(subdir, file)
                        update_csv_remove_rows_before_last_step_num_negative_one(attack_x_y_file_path)

if __name__ == '__main__':
    start_directory = '/home/borito1907/impossibility-watermark/results/stationary_distribution/robustness_analysis'
    process_corpuses(start_directory)
