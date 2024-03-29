import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from diversity_oracle import DiversityOracle
from utils import save_to_csv

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


def main(rewrite=True):
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process diversity analysis parameters.')
    parser.add_argument('entropy', type=int, help='Entropy value for analysis.')
    parser.add_argument('output', type=int, help='Output value for analysis.')
    # Add rewrite as a keyword argument, defaulting to False
    parser.add_argument('--rewrite', action='store_true', help='Enable rewriting existing files.')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Use arguments
    entropy = args.entropy
    output = args.output
    rewrite = args.rewrite
    
    root_dir = f"results/stationary_distribution/robustness_analysis/entropy_{entropy}/output_{output}"
    csv_file_directory = f"{root_dir}/corpuses"
    diversity_directory = f"{root_dir}/diversity"

    for csv_filename in os.listdir(csv_file_directory):
        print(f"Filename: {csv_filename}")
        csv_filepath = os.path.join(csv_file_directory, csv_filename)
        # Remove the .csv at the end of the filename
        attack_id = csv_filename[:-4]
        attack_diversity_folder = f"{diversity_directory}/{attack_id}"

        # Check if the CSV file already exists
        if not rewrite and os.path.exists(attack_diversity_folder):
            print(f"The folder for {attack_id} already exists. Skipping.")
            continue
        
        os.makedirs(attack_diversity_folder, exist_ok=True)
        
        csv_files = [csv_filepath]
        
        corpus_df = get_diversity_df(csv_files, normalized=False)
        individual_df = get_diversity_df(csv_files, normalized=True)  
        
        save_to_csv(individual_df, attack_diversity_folder, 'individual.csv', rewrite=True)  
        save_plots(individual_df, attack_diversity_folder)  

        save_to_csv(corpus_df, attack_diversity_folder, 'corpus.csv', rewrite=True)  
        save_plots(corpus_df, attack_diversity_folder)  
        
        
if __name__ == "__main__":
    main()