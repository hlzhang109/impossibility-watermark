import re
import pandas as pd
import os
import json

def save_to_csv(data, filename):
    df_out = pd.DataFrame(data)
    if os.path.exists(filename):
        df_out.to_csv(filename, mode='a', header=False, index=False)  # Append without writing headers
    else:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_out.to_csv(filename, index=False)  # Create new file with headers
    print(f"Data appended to {filename}")
    
def load_data(filename):
    """Load JSON data from a file."""
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)
    
# Read the content of the text file
def find_csv(txt_file_path):
    with open(txt_file_path, 'r') as file:
        content = file.read()

    # Search for the first occurrence of 'attack_*.csv' pattern
    match = re.search(r'attack_.*\.csv', content)

    if match:
        csv_filename = match.group(0)
        return csv_filename
    return None

def count_words(text):
    if text is None:
        return 0
    return len(text.split())