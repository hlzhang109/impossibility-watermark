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