import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from diversity_oracle import DiversityOracle
from utils import save_to_csv, find_csv, count_words

div_oracle = DiversityOracle(metrics = {}, verbose=False, normalized=False)
normalized_div_oracle = DiversityOracle(metrics = {}, verbose=False, normalized=True)

def get_success_dfs(csv_files):
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df[df['quality_preserved'] == True])
    return dfs

def create_corpuses(dfs, normalized):
    corpuses = []
    min_length = min(len(df) for df in dfs)
    
    if normalized:
        for i in range(min_length):
            corpus = [df.iloc[i]['mutated_text'] for df in dfs]
            corpuses.append(corpus)
    else:
        for i in range(min_length):
            corpus = [text for df in dfs for text in df.iloc[:i+1]['mutated_text']]
            corpuses.append(corpus)
    return corpuses

def get_diversity_df(csv_files, normalized):
    dfs = get_success_dfs(csv_files)
    corpuses = create_corpuses(dfs, normalized)
    metric_dicts = []
    for corpus in corpuses:
        if normalized:
            metrics = normalized_div_oracle(corpus)
        else:
            metrics = div_oracle(corpus)
        metric_dict = {metric['metric_name']: metric['diversity_score'] for metric in metrics}
        
        metric_dicts.append(metric_dict)
    
    df = pd.DataFrame(metric_dicts)
    return df

def plot_metric(df, column_name):

    # Plotting
    plt.figure(figsize=(10, 6))  # Set the figure size (optional)
    plt.plot(df.index, df[column_name], marker='o', linestyle='-', color='b')  # Plot with line and markers
    plt.title(f"Evolution of {column_name}")  # Title of the plot
    plt.xlabel('Step Number')  # X-axis label
    plt.ylabel(column_name)  # Y-axis label, replace with your column name
    plt.grid(True)  # Show grid
    plt.show()

def save_plots(df, folder):
    for column_name in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[column_name], marker='o', linestyle='-', color='b')
        plt.title(f'Evolution of {column_name} Values Over Rows')
        plt.xlabel('Step Number')
        plt.ylabel(column_name)
        plt.grid(True)
        
        # Save the figure as a PNG file
        filename = f'{folder}/{column_name}.png'
        plt.savefig(filename, dpi=300)
        
        # Clear the current figure to avoid overlapping of plots
        plt.clf()  # Use plt.close() if you want to close the figure completely


def main():
    # Non-normalized Values
    attack_type = '4'
    txt_file_directory = f"./fourth_round/"
    plots_folder = f"./plots/plots_{attack_type}/"

    # TODO: Refactor the txt file stuff.
    for txt_filename in os.listdir(txt_file_directory):
        print(f"Filename: {txt_filename}")
        txt_filepath = os.path.join(txt_file_directory, txt_filename)
        csv_filename = find_csv(txt_filepath)
        csv_filename = txt_filename[:-4] + '.csv'
        directory = "./eval/results/"
        csv_filepath = os.path.join(directory, csv_filename)
    

        directory_path = os.path.join(plots_folder, txt_filename[:-4])
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        
        csv_path = os.path.join(directory_path, 'div_df.csv')
        
        # Check if the CSV file already exists
        if os.path.exists(csv_path):
            print(f"CSV file {csv_path} already exists. Skipping.")
            continue  # Skip the rest of the loop and proceed with the next txt_filename
        
        csv_files = [csv_filepath]
        
        div_df = get_diversity_df(csv_files, False)  
        
        save_to_csv(div_df, csv_path, rewrite=True)  
        save_plots(div_df, directory_path)  

        
        
if __name__ == "__main__":
    main()