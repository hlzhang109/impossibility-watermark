import os
import pandas as pd

def compare_attack_files(attack_x_file_path, attack_x_y_file_path):
    df_x = pd.read_csv(attack_x_file_path)
    df_x_y = pd.read_csv(attack_x_y_file_path)

    df_x = df_x.drop('timestamp', axis=1, errors='ignore')
    df_x_y = df_x_y.drop('timestamp', axis=1, errors='ignore')

    common_columns = set(df_x.columns) & set(df_x_y.columns)
    df_x = df_x[list(common_columns)]
    df_x_y = df_x_y[list(common_columns)]

    row_diff_found = not df_x.equals(df_x_y.iloc[:len(df_x)])
    
    diff_details = None
    current_text_diff_found = False  # Initialize as False

    if row_diff_found:
        current_text_diff_found = df_x['current_text'].iloc[-1] != df_x_y['current_text'].iloc[0]
        
        for i, (index_x, row_x) in enumerate(df_x.iterrows()):
            row_x_y = df_x_y.iloc[i]
            if not row_x.equals(row_x_y):
                diff_columns = row_x[row_x != row_x_y].index.tolist()
                diff_values_x = row_x[diff_columns].to_dict()
                diff_values_x_y = row_x_y[diff_columns].to_dict()
                diff_details = {
                    'row_index': i,
                    'diff_columns': diff_columns,
                    'values_in_x': diff_values_x,
                    'values_in_x_y': diff_values_x_y
                }
                break  # Stop after finding the first difference

    # Return flags and details of the first difference
    return row_diff_found, current_text_diff_found, diff_details

def process_corpuses(start_path):
    for subdir, dirs, files in os.walk(start_path):
        if os.path.basename(subdir) == 'corpuses':
            for file in files:
                if file.startswith('attack_') and file.endswith('.csv'):
                    parts = file.split('_')
                    if len(parts) == 3:
                        attack_x_y_file_path = os.path.join(subdir, file)
                        attack_x_file_path = os.path.join(subdir, f"attack_{parts[1]}.csv")
                        if os.path.exists(attack_x_file_path):
                            row_diff_found, current_text_diff_found, diff_details = compare_attack_files(attack_x_file_path, attack_x_y_file_path)
                            if row_diff_found:
                                print('=' * 100)
                                print(f"Differences found for files: {attack_x_file_path} and {attack_x_y_file_path}")
                                if current_text_diff_found:
                                    print("Difference in 'current_text' found.")
                                if diff_details:
                                    print(f"First difference at row {diff_details['row_index']}:")
                                    for col in diff_details['diff_columns']:
                                        print(f"Column '{col}' differs: {diff_details['values_in_x'][col]} (in {attack_x_file_path}) != {diff_details['values_in_x_y'][col]} (in {attack_x_y_file_path})")

if __name__ == '__main__':
    start_directory = '/home/borito1907/impossibility-watermark/results/stationary_distribution/robustness_analysis'
    process_corpuses(start_directory)
